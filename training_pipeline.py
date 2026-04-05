"""
Multi-Stage Transfer Learning Training Pipeline for Medical Imaging

Three-stage approach:
1. ImageNet pretrained model
2. Domain training on public datasets
3. Fine-tuning on clinical dataset
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import json
from datetime import datetime

from sklearn.metrics import roc_auc_score


@dataclass
class TrainingConfig:
    """Training configuration for multi-stage pipeline."""
    
    # Phase A: Domain training
    domain_learning_rate: float = 0.001
    domain_epochs: int = 50
    domain_batch_size: int = 32
    domain_optimizer: str = "adam"  # adam, sgd
    domain_weight_decay: float = 1e-4
    
    # Phase B: Clinical fine-tuning
    clinical_learning_rate: float = 0.0001
    clinical_epochs: int = 30
    clinical_batch_size: int = 16
    clinical_optimizer: str = "adam"
    clinical_weight_decay: float = 1e-4
    freeze_backbone: bool = False
    freeze_until_layer: Optional[int] = None  # Freeze layers 0 to N
    
    # Common
    model_type: str = "resnet18"  # resnet18, efficientnet_b0
    num_classes: int = 2
    dropout_rate: float = 0.5
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    patience: int = 5  # Early stopping patience
    # Metric monitored for early stopping and best-model selection.
    # Phase A uses phase_a_monitor_metric; Phase B uses monitor_metric.
    # Supported values: 'accuracy', 'recall', 'f1', 'auc'
    monitor_metric: str = 'recall'           # Phase B (clinical) default: maximise recall
    phase_a_monitor_metric: str = 'accuracy' # Phase A (domain) default: accuracy


class BinaryClassificationModel(nn.Module):
    """Binary classification model with backbone and custom head."""

    def __init__(self, model_type: str = "resnet18", pretrained: bool = True,
                 dropout_rate: float = 0.5):
        super().__init__()

        self.model_type = model_type

        if model_type == "resnet18":
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = resnet18(weights=weights)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif model_type == "efficientnet_b0":
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()

        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Custom classification head
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 2)  # Binary classification
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.backbone(x)
        logits = self.head(features)
        return logits


class TrainingMetrics:
    """Track training metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.losses = []
        self.accuracies = []
        self.recalls = []
        self.precisions = []
        self.f1_scores = []
    
    def add_batch(self, loss: float, accuracy: float, recall: float, 
                  precision: float, f1: float):
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        self.recalls.append(recall)
        self.precisions.append(precision)
        self.f1_scores.append(f1)
    
    def get_averages(self) -> Dict[str, float]:
        return {
            'loss': np.mean(self.losses),
            'accuracy': np.mean(self.accuracies),
            'recall': np.mean(self.recalls),
            'precision': np.mean(self.precisions),
            'f1': np.mean(self.f1_scores)
        }


def compute_class_weights(dataloader: DataLoader, device: str) -> torch.Tensor:
    """
    Compute inverse-frequency class weights from training labels.

    Clinical datasets are typically imbalanced (more benign than malignant).
    Passing these weights to CrossEntropyLoss increases sensitivity — the
    most safety-critical metric for cancer detection — without discarding data.

    Returns:
        Tensor of shape (num_classes,) on the given device.
    """
    all_labels: List[int] = []
    for batch in dataloader:
        _, labels, *_ = batch
        all_labels.extend(labels.numpy().tolist())

    labels_arr = np.array(all_labels)
    counts = np.bincount(labels_arr, minlength=2).astype(float)
    # Inverse-frequency: rarer class gets higher weight
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * len(counts)   # normalise so mean weight = 1
    print(f"  Class distribution: {counts.astype(int)}, weights: {weights.round(3)}")
    return torch.tensor(weights, dtype=torch.float32).to(device)


def compute_metrics(outputs: torch.Tensor, targets: torch.Tensor) -> Tuple[float, float, float, float]:
    """Compute accuracy, recall, precision, and F1 score."""
    
    predictions = torch.argmax(outputs, dim=1)
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    # Binary classification metrics
    tp = np.sum((predictions == 1) & (targets == 1))
    tn = np.sum((predictions == 0) & (targets == 0))
    fp = np.sum((predictions == 1) & (targets == 0))
    fn = np.sum((predictions == 0) & (targets == 1))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return accuracy, recall, precision, f1


def get_optimizer(model: nn.Module, learning_rate: float, optimizer_type: str,
                  weight_decay: float) -> torch.optim.Optimizer:
    """Create optimizer."""
    
    if optimizer_type.lower() == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type.lower() == "sgd":
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, 
                        weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def freeze_backbone(model: BinaryClassificationModel, freeze_until_layer: Optional[int] = None):
    """Freeze backbone layers for fine-tuning."""
    
    if isinstance(model.backbone, nn.Module):
        layers = list(model.backbone.named_modules())
        
        if freeze_until_layer is not None:
            # Freeze layers 0 to freeze_until_layer
            for idx, (name, module) in enumerate(layers):
                if idx <= freeze_until_layer:
                    for param in module.parameters():
                        param.requires_grad = False
        else:
            # Freeze entire backbone
            for param in model.backbone.parameters():
                param.requires_grad = False


def train_epoch(model: BinaryClassificationModel, dataloader: DataLoader,
                criterion: nn.Module, optimizer: torch.optim.Optimizer,
                device: str) -> Dict[str, float]:
    """Train for one epoch. Metrics computed globally, not averaged per batch."""

    model.train()
    total_loss = 0.0
    # FIX (batch-average metrics): accumulate all predictions and targets;
    # compute recall/F1/precision once over the full epoch, not as a batch average.
    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[float] = []   # softmax P(positive) for AUC

    for images, labels, *_ in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy().tolist())
        all_targets.extend(labels.cpu().numpy().tolist())
        all_probs.extend(torch.softmax(outputs.detach(), dim=1)[:, 1].cpu().numpy().tolist())

    accuracy, recall, precision, f1 = compute_metrics(
        torch.tensor(all_preds), torch.tensor(all_targets)
    )
    auc: Optional[float] = None
    if len(set(all_targets)) > 1:
        try:
            auc = float(roc_auc_score(all_targets, all_probs))
        except Exception:
            pass

    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'auc': auc,
    }


def validate_epoch(model: BinaryClassificationModel, dataloader: DataLoader,
                   criterion: nn.Module, device: str) -> Dict[str, float]:
    """Validate for one epoch. Metrics computed globally, not averaged per batch."""

    model.eval()
    total_loss = 0.0
    # FIX (batch-average metrics): same fix as train_epoch — accumulate then compute.
    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[float] = []   # softmax P(positive) for AUC

    with torch.no_grad():
        for images, labels, *_ in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy().tolist())

    accuracy, recall, precision, f1 = compute_metrics(
        torch.tensor(all_preds), torch.tensor(all_targets)
    )
    auc: Optional[float] = None
    if len(set(all_targets)) > 1:
        try:
            auc = float(roc_auc_score(all_targets, all_probs))
        except Exception:
            pass

    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'auc': auc,
    }


def save_checkpoint(model: BinaryClassificationModel, optimizer: torch.optim.Optimizer,
                   epoch: int, metrics: Dict[str, float], checkpoint_path: str):
    """Save model checkpoint."""
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Checkpoint saved: {checkpoint_path}")


def load_checkpoint(model: BinaryClassificationModel, checkpoint_path: str,
                   device: str) -> Dict:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Checkpoint loaded: {checkpoint_path}")
    return checkpoint


def phase_a_domain_training(model: BinaryClassificationModel, 
                           train_dataloader: DataLoader,
                           val_dataloader: DataLoader,
                           config: TrainingConfig) -> str:
    """
    Phase A: Domain training on public datasets.
    
    Train on public data to learn domain-specific features.
    Save checkpoint_domain.pth for Phase B fine-tuning.
    """
    
    print("\n" + "="*80)
    print("PHASE A: DOMAIN TRAINING")
    print("="*80)
    print(f"Training on public datasets for {config.domain_epochs} epochs")
    print(f"Learning rate: {config.domain_learning_rate}")
    print(f"Optimizer: {config.domain_optimizer}")
    
    device = config.device
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, config.domain_learning_rate,
                             config.domain_optimizer, config.domain_weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.domain_epochs)

    # FIX (class imbalance): compute class weights from the training split.
    # Public datasets may also be class-imbalanced; weighted loss improves sensitivity.
    print("  Computing class weights for domain training...")
    class_weights = compute_class_weights(train_dataloader, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    best_val_loss = float('inf')
    best_val_metric = 0.0
    patience_counter = 0
    best_checkpoint_path = os.path.join(config.checkpoint_dir, "phase_a_best.pth")
    monitor = config.phase_a_monitor_metric
    print(f"  Monitoring: {monitor} (early stopping)")
    
    for epoch in range(config.domain_epochs):
        # Train
        train_metrics = train_epoch(model, train_dataloader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate_epoch(model, val_dataloader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step()
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{config.domain_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, Recall: {train_metrics['recall']:.4f}, AUC: {train_metrics['auc']}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, Recall: {val_metrics['recall']:.4f}, AUC: {val_metrics['auc']}")
        
        # Save best model based on monitored metric
        monitored = val_metrics.get(monitor) or 0.0
        if monitored > best_val_metric:
            best_val_metric = monitored
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_metrics, best_checkpoint_path)
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch+1} (best {monitor}={best_val_metric:.4f})")
            break
    
    # Save final domain checkpoint
    domain_checkpoint_path = os.path.join(config.checkpoint_dir, "checkpoint_domain.pth")
    save_checkpoint(model, optimizer, config.domain_epochs, val_metrics, domain_checkpoint_path)
    
    print(f"\n\u2713 Phase A complete - Best val {config.phase_a_monitor_metric}: {best_val_metric:.4f}")
    return domain_checkpoint_path


def phase_b_clinical_finetuning(model: BinaryClassificationModel,
                               domain_checkpoint_path: str,
                               train_dataloader: DataLoader,
                               val_dataloader: DataLoader,
                               test_dataloader: DataLoader,
                               config: TrainingConfig) -> Tuple[str, Dict]:
    """
    Phase B: Clinical fine-tuning.
    
    Load domain weights from Phase A.
    Fine-tune on clinical dataset with lower learning rate.
    Optionally freeze early layers.
    """
    
    print("\n" + "="*80)
    print("PHASE B: CLINICAL FINE-TUNING")
    print("="*80)
    
    device = config.device
    
    # Load domain checkpoint
    print(f"Loading domain checkpoint: {domain_checkpoint_path}")
    load_checkpoint(model, domain_checkpoint_path, device)
    
    # Optionally freeze backbone
    if config.freeze_backbone or config.freeze_until_layer is not None:
        print("Freezing backbone layers...")
        freeze_backbone(model, config.freeze_until_layer)
    
    print(f"Fine-tuning on clinical dataset for {config.clinical_epochs} epochs")
    print(f"Learning rate: {config.clinical_learning_rate}")
    print(f"Optimizer: {config.clinical_optimizer}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, config.clinical_learning_rate,
                             config.clinical_optimizer, config.clinical_weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.clinical_epochs)

    # FIX (class imbalance): weight loss by inverse class frequency on clinical train set.
    # Clinical data is often imbalanced; prioritising the minority (malignant) class
    # increases sensitivity, which is the most safety-critical metric.
    print("  Computing class weights for clinical fine-tuning...")
    class_weights = compute_class_weights(train_dataloader, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    best_val_metric = 0.0
    patience_counter = 0
    best_checkpoint_path = os.path.join(config.checkpoint_dir, "phase_b_best.pth")
    monitor = config.monitor_metric
    print(f"  Monitoring: {monitor} (early stopping — clinical fine-tuning)")

    for epoch in range(config.clinical_epochs):
        # Train
        train_metrics = train_epoch(model, train_dataloader, criterion, optimizer, device)

        # Validate
        val_metrics = validate_epoch(model, val_dataloader, criterion, device)

        # FIX (test leakage): test set is NOT evaluated here.

        # Learning rate scheduling
        scheduler.step()

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{config.clinical_epochs}")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, Recall: {train_metrics['recall']:.4f}, AUC: {train_metrics['auc']}")
            print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, Recall: {val_metrics['recall']:.4f}, AUC: {val_metrics['auc']}")

        # Save best model based on monitored metric
        monitored = val_metrics.get(monitor) or 0.0
        if monitored > best_val_metric:
            best_val_metric = monitored
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_metrics, best_checkpoint_path)
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch+1} (best {monitor}={best_val_metric:.4f})")
            break

    # Save final clinical checkpoint
    clinical_checkpoint_path = os.path.join(config.checkpoint_dir, "checkpoint_clinical.pth")
    save_checkpoint(model, optimizer, config.clinical_epochs, val_metrics, clinical_checkpoint_path)

    # FIX (test leakage): evaluate test set exactly once, using the best val checkpoint.
    load_checkpoint(model, best_checkpoint_path, device)
    test_metrics = validate_epoch(model, test_dataloader, criterion, device)

    print(f"\n✓ Phase B complete - Best val {config.monitor_metric}: {best_val_metric:.4f}")
    print(f"  Final test metrics (best val checkpoint):")
    print(f"    Accuracy: {test_metrics.get('accuracy', 0.0):.4f}")
    print(f"    Recall:   {test_metrics.get('recall', 0.0):.4f}")
    print(f"    AUC:      {test_metrics.get('auc', 'N/A')}")

    return clinical_checkpoint_path, test_metrics


def run_training_pipeline(config: TrainingConfig, 
                         domain_train_loader: DataLoader,
                         domain_val_loader: DataLoader,
                         clinical_train_loader: DataLoader,
                         clinical_val_loader: DataLoader,
                         clinical_test_loader: DataLoader):
    """
    Run complete multi-stage training pipeline.
    
    Args:
        config: Training configuration
        domain_train_loader: DataLoader for domain training set
        domain_val_loader: DataLoader for domain validation set
        clinical_train_loader: DataLoader for clinical training set
        clinical_val_loader: DataLoader for clinical validation set
        clinical_test_loader: DataLoader for clinical test set
    """
    
    # FIX (reproducibility): set all seeds for full determinism on CPU and GPU.
    # Without cudnn.deterministic, CUDA convolutions remain non-deterministic even
    # with torch.manual_seed, producing different results across runs.
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False   # benchmark=True disables determinism
    
    device = config.device
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    print(f"\nInitializing {config.model_type} model...")
    model = BinaryClassificationModel(
        model_type=config.model_type,
        pretrained=True,
        dropout_rate=config.dropout_rate
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Phase A: Domain training
    domain_checkpoint = phase_a_domain_training(
        model, domain_train_loader, domain_val_loader, config
    )
    
    # Load best domain model
    load_checkpoint(model, domain_checkpoint, device)
    
    # Phase B: Clinical fine-tuning
    clinical_checkpoint, test_metrics = phase_b_clinical_finetuning(
        model, domain_checkpoint, 
        clinical_train_loader, clinical_val_loader, clinical_test_loader,
        config
    )
    
    # Save training summary
    summary = {
        'config': {
            'model_type': config.model_type,
            'domain_learning_rate': config.domain_learning_rate,
            'domain_epochs': config.domain_epochs,
            'clinical_learning_rate': config.clinical_learning_rate,
            'clinical_epochs': config.clinical_epochs,
            'freeze_backbone': config.freeze_backbone,
        },
        'phase_a_checkpoint': domain_checkpoint,
        'phase_b_checkpoint': clinical_checkpoint,
        'test_metrics': test_metrics,
        'timestamp': datetime.now().isoformat()
    }
    
    summary_path = os.path.join(config.checkpoint_dir, "training_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("TRAINING PIPELINE COMPLETE")
    print("="*80)
    print(f"Domain checkpoint: {domain_checkpoint}")
    print(f"Clinical checkpoint: {clinical_checkpoint}")
    print(f"Summary saved: {summary_path}")
    
    return model, summary


# Example usage
if __name__ == "__main__":
    """
    Example training pipeline execution.
    
    Note: In practice, you would:
    1. Load actual datasets using data_loader.py and dataset_split.py
    2. Pass DataLoaders to run_training_pipeline()
    
    Example:
        from data_loader import get_clinical_dataset, get_public_dataset
        from dataset_split import get_clinical_dataloaders, get_public_dataloaders
        
        config = TrainingConfig(
            model_type="resnet18",
            domain_learning_rate=0.001,
            clinical_learning_rate=0.0001,
            freeze_backbone=True
        )
        
        # Load datasets
        public_dataset = get_public_dataset("data/public_dataset_1")
        clinical_dataset = get_clinical_dataset("data/clinical_dataset")
        
        # Create dataloaders
        public_loaders = get_public_dataloaders(public_dataset, batch_size=32)
        clinical_loaders = get_clinical_dataloaders(clinical_dataset, batch_size=16)
        
        # Run pipeline
        model, summary = run_training_pipeline(
            config,
            domain_train_loader=public_loaders['train'],
            domain_val_loader=public_loaders['val'],
            clinical_train_loader=clinical_loaders['train'],
            clinical_val_loader=clinical_loaders['val'],
            clinical_test_loader=clinical_loaders['test']
        )
    """
    
    # Create training config
    config = TrainingConfig(
        model_type="resnet18",
        domain_learning_rate=0.001,
        domain_epochs=50,
        clinical_learning_rate=0.0001,
        clinical_epochs=30,
        freeze_backbone=True,
        checkpoint_dir="./checkpoints",
        seed=42
    )
    
    print("Training pipeline module loaded successfully!")
    print("See example usage comments at bottom of file for integration with data loaders.")
