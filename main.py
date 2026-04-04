"""
main.py — Clinical AI Pipeline Entry Point

Runnable end-to-end: load → split → train → evaluate.

Usage
-----
  python main.py                        # public-only mode, real data
  python main.py --debug                # synthetic data, 60 samples, 3 epochs
  python main.py --clinical data/my_clinical  # include clinical dataset
  python main.py --mode public_only     # force public-only even if clinical exists
  python main.py --epochs 30 --lr 5e-4  --batch-size 16

Modes
-----
  public_only : train+val on public data; no test set
  clinical    : public data for pretraining (Phase A),
                clinical data for fine-tuning and test (Phase B)
                Clinical dataset is NEVER mixed into training.
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

# ── project modules ────────────────────────────────────────────────────────
from data_loader import ClinicalDataset, PublicDataset
from preprocessing import (
    AugmentationConfig,
    TransformPipeline,
    TransformedDataset,
    compute_dataset_stats,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from dataset_split import DataLoaderFactory, SplitConfig
from training_pipeline import (
    BinaryClassificationModel,
    TrainingConfig,
    train_epoch,
    validate_epoch,
    compute_class_weights,
    load_checkpoint,
    phase_a_domain_training,
)
from evaluation import EvaluationConfig
from evaluation import evaluate as _full_evaluate


logger = logging.getLogger(__name__)


# =============================================================================
# CONFIG  —  edit these defaults or override via CLI flags
# =============================================================================

DEFAULT_CONFIG = {
    # Paths
    "public_dataset_path":   "data/public_dataset_1",
    "clinical_dataset_path": "data/clinical_dataset",
    # Training
    "batch_size":   32,
    "learning_rate": 1e-4,
    "epochs":       50,
    "patience":     7,       # early-stopping patience (epochs without val improvement)
    # Model
    "model":        "resnet18",
    "dropout":      0.3,
    # Reproducibility
    "seed":         42,
    # Hardware
    "num_workers":  0,       # set 0 on Windows to avoid DataLoader issues
    # Outputs
    "checkpoint_dir": "checkpoints",
}


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# SYNTHETIC DATASET  (debug mode — no real data needed)
# =============================================================================

class SyntheticDataset(Dataset):
    """In-memory dataset of random PIL images. Used in --debug mode only."""

    def __init__(self, n: int = 60, seed: int = 42):
        rng = np.random.default_rng(seed)
        self._images = [
            Image.fromarray(rng.integers(0, 255, (224, 224, 3), dtype=np.uint8))
            for _ in range(n)
        ]
        self._labels = [int(i % 2) for i in range(n)]   # balanced 0/1

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int):
        meta = {"image_id": str(idx), "image_name": f"synthetic_{idx}.png",
                "dataset_source": "SYNTHETIC", "clinician_diagnosis": None}
        return self._images[idx], self._labels[idx], meta


# =============================================================================
# TRAINING HELPERS  (delegate to training_pipeline)
# =============================================================================

def _make_training_config(cfg: dict, monitor_metric: str) -> TrainingConfig:
    """Build a TrainingConfig from the flat cfg dict and the CLI monitor arg."""
    return TrainingConfig(
        domain_learning_rate=cfg["learning_rate"],
        domain_epochs=cfg["epochs"],
        domain_batch_size=cfg["batch_size"],
        domain_weight_decay=1e-4,
        clinical_learning_rate=cfg["learning_rate"],
        clinical_epochs=cfg["epochs"],
        clinical_batch_size=cfg["batch_size"],
        clinical_weight_decay=1e-4,
        checkpoint_dir=cfg["checkpoint_dir"],
        seed=cfg["seed"],
        num_workers=cfg["num_workers"],
        patience=cfg["patience"],
        monitor_metric=monitor_metric,
        phase_a_monitor_metric="accuracy",
    )


def _train_single_phase(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    train_cfg: TrainingConfig,
    device: str,
) -> nn.Module:
    """
    Single-phase training loop with early stopping on train_cfg.monitor_metric.

    Used for debug mode, public-only mode, and clinical mode when no public
    pre-training data is available.  Delegates epoch logic to train_epoch /
    validate_epoch from training_pipeline so the metric computation is identical
    to the two-phase pipeline.
    """
    criterion = nn.CrossEntropyLoss(
        weight=compute_class_weights(loaders["train"], device)
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.domain_learning_rate,
        weight_decay=train_cfg.domain_weight_decay,
    )
    monitor = train_cfg.monitor_metric
    best_val_metric = 0.0
    patience_counter = 0
    ckpt_path = Path(train_cfg.checkpoint_dir) / "best_model.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, train_cfg.domain_epochs + 1):
        tr = train_epoch(model, loaders["train"], criterion, optimizer, device)
        va = validate_epoch(model, loaders["val"], criterion, device)

        print(
            f"Epoch {epoch:3d}/{train_cfg.domain_epochs}  "
            f"loss={tr['loss']:.4f}  acc={tr['accuracy']:.3f}  "
            f"recall={tr['recall']:.3f}  |  "
            f"val_loss={va['loss']:.4f}  val_acc={va['accuracy']:.3f}  "
            f"val_{monitor}={va.get(monitor) or 0.0:.3f}"
        )

        monitored = va.get(monitor) or 0.0
        if monitored > best_val_metric:
            best_val_metric = monitored
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= train_cfg.patience:
                print(
                    f"  Early stopping at epoch {epoch}  "
                    f"(best val_{monitor}={best_val_metric:.3f})"
                )
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"\n✓ Best checkpoint loaded from {ckpt_path}")
    return model


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_public_dataset(path: str) -> Optional[PublicDataset]:
    """Load a public dataset; return None with a clear message if unavailable."""
    p = Path(path)
    if not p.exists():
        print(f"  [WARN] Public dataset not found: {p}  — skipping.")
        return None
    images_dir = p / "images"
    if not images_dir.exists() or not any(images_dir.iterdir()):
        print(f"  [WARN] No images found in {images_dir}  — skipping.")
        return None
    try:
        ds = PublicDataset(root_dir=p)
        print(f"  Loaded public dataset: {len(ds)} samples  ({p})")
        return ds
    except Exception as e:
        print(f"  [WARN] Could not load public dataset at {p}: {e}")
        return None


def load_clinical_dataset(path: str) -> Optional[ClinicalDataset]:
    """Load clinical dataset; return None with a clear message if unavailable."""
    p = Path(path)
    if not p.exists():
        print(f"  [INFO] Clinical dataset path not found: {p}")
        print("         Running in public-only mode.")
        return None
    images_dir = p / "images"
    if not images_dir.exists() or not any(images_dir.iterdir()):
        print(f"  [WARN] No images found in {images_dir}. Skipping clinical dataset.")
        return None
    try:
        ds = ClinicalDataset(root_dir=p)
        print(f"  Loaded clinical dataset: {len(ds)} samples  ({p})")
        return ds
    except Exception as e:
        print(f"  [WARN] Could not load clinical dataset at {p}: {e}")
        return None


# =============================================================================
# DATALOADER HELPERS
# =============================================================================

def _make_transforms(
    normalization_mean: Optional[List[float]] = None,
    normalization_std: Optional[List[float]] = None,
):
    """Return a dict of {train, val, test} transforms with optional custom stats."""
    pipeline = TransformPipeline(
        augmentation_config=AugmentationConfig(),
        normalization_mean=normalization_mean or IMAGENET_MEAN,
        normalization_std=normalization_std or IMAGENET_STD,
    )
    return pipeline.get_transforms_dict()


def _resolve_normalization_stats(dataset, train_indices: List[int]) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Compute dataset-specific normalization on the training subset when feasible.

    Falls back to ImageNet statistics when the training split is too small
    (< 50 images) or if statistic computation fails.
    """
    if len(train_indices) < 50:
        logger.warning(
            "Training split too small for dataset-specific normalization (%d < 50). "
            "Falling back to ImageNet statistics.",
            len(train_indices),
        )
        return None, None

    try:
        mean, std = compute_dataset_stats(Subset(dataset, train_indices), sample_size=500)
        logger.info("Using dataset-specific normalization stats: mean=%s std=%s", mean, std)
        return mean, std
    except Exception as exc:
        logger.warning(
            "Failed to compute dataset-specific normalization stats (%s). "
            "Falling back to ImageNet statistics.",
            exc,
        )
        return None, None


def _subset_loader(dataset, indices, transform, batch_size, num_workers,
                   shuffle=False) -> DataLoader:
    td = TransformedDataset(Subset(dataset, indices), transform)
    return DataLoader(td, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)


def build_public_only_loaders(dataset, cfg: dict) -> Dict[str, DataLoader]:
    """
    Split public dataset into train/val/test using SplitConfig ratios.

    This avoids a second hard-coded source of truth for three-way splits.
    """
    from sklearn.model_selection import train_test_split as _tts
    split_cfg = SplitConfig(random_seed=cfg["seed"])
    labels = np.array([m.label for _, m in dataset.samples])
    indices = np.arange(len(dataset))

    # Step 1: carve out the held-out test split
    trainval_idx, test_idx = _tts(
        indices,
        test_size=split_cfg.clinical_test_ratio,
        stratify=labels,
        random_state=cfg["seed"],
    )
    # Step 2: split remaining pool into train/val with the same SplitConfig ratios
    trainval_labels = labels[trainval_idx]
    train_ratio = split_cfg.clinical_train_ratio / (
        split_cfg.clinical_train_ratio + split_cfg.clinical_val_ratio
    )
    train_idx, val_idx = _tts(
        trainval_idx,
        train_size=train_ratio,
        stratify=trainval_labels,
        random_state=cfg["seed"],
    )

    # Verify no leakage
    assert not (set(train_idx) & set(val_idx))
    assert not (set(train_idx) & set(test_idx))
    assert not (set(val_idx)   & set(test_idx))

    mean, std = _resolve_normalization_stats(dataset, train_idx.tolist())
    tf = _make_transforms(mean, std)
    bs, nw = cfg["batch_size"], cfg["num_workers"]
    return {
        "train": _subset_loader(dataset, train_idx.tolist(), tf["train"], bs, nw, shuffle=True),
        "val":   _subset_loader(dataset, val_idx.tolist(),   tf["val"],   bs, nw),
        "test":  _subset_loader(dataset, test_idx.tolist(),  tf["test"],  bs, nw),
    }


def build_clinical_loaders(clinical_ds, cfg: dict) -> Dict[str, DataLoader]:
    """Split clinical dataset 70/15/15 into proper train/val/test loaders."""
    split_cfg = SplitConfig(random_seed=cfg["seed"])
    from dataset_split import split_clinical_dataset
    splits = split_clinical_dataset(clinical_ds, split_cfg)

    tf = _make_transforms()
    bs, nw = cfg["batch_size"], cfg["num_workers"]
    return {
        "train": _subset_loader(clinical_ds, splits["train"], tf["train"], bs, nw, shuffle=True),
        "val":   _subset_loader(clinical_ds, splits["val"],   tf["val"],   bs, nw),
        "test":  _subset_loader(clinical_ds, splits["test"],  tf["test"],  bs, nw),
    }


def build_debug_loaders(cfg: dict, n_samples: int = 60) -> Dict[str, DataLoader]:
    """
    Build loaders from synthetic data — no real datasets required.
    Useful for quickly checking the whole pipeline is wired correctly.
    """
    ds = SyntheticDataset(n=n_samples, seed=cfg["seed"])
    n = len(ds)
    split_cfg = SplitConfig(random_seed=cfg["seed"])
    n_train = int(n * split_cfg.clinical_train_ratio)
    n_val   = int(n * split_cfg.clinical_val_ratio)
    idx = list(range(n))
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    tf = _make_transforms()
    bs, nw = cfg["batch_size"], cfg["num_workers"]
    return {
        "train": _subset_loader(ds, train_idx, tf["train"], bs, nw, shuffle=True),
        "val":   _subset_loader(ds, val_idx,   tf["val"],   bs, nw),
        "test":  _subset_loader(ds, test_idx,  tf["test"],  bs, nw),
    }


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Clinical AI — Oral Lesion Classifier")
    p.add_argument("--debug", action="store_true",
                   help="Run on synthetic data (no real dataset needed). "
                        "Uses 60 samples and 3 epochs.")
    p.add_argument("--mode", choices=["public_only", "clinical"], default=None,
                   help="Force a specific mode. Default: auto-detect from available data.")
    p.add_argument("--public",   default=DEFAULT_CONFIG["public_dataset_path"],
                   metavar="PATH", help="Path to public dataset directory.")
    p.add_argument("--clinical", default=DEFAULT_CONFIG["clinical_dataset_path"],
                   metavar="PATH", help="Path to clinical dataset directory.")
    p.add_argument("--epochs",     type=int,   default=DEFAULT_CONFIG["epochs"])
    p.add_argument("--lr",         type=float, default=DEFAULT_CONFIG["learning_rate"],
                   dest="learning_rate")
    p.add_argument("--batch-size", type=int,   default=DEFAULT_CONFIG["batch_size"],
                   dest="batch_size")
    p.add_argument("--patience",   type=int,   default=DEFAULT_CONFIG["patience"])
    p.add_argument("--seed",       type=int,   default=DEFAULT_CONFIG["seed"])
    p.add_argument("--num-workers", type=int,  default=DEFAULT_CONFIG["num_workers"],
                   dest="num_workers")
    p.add_argument("--checkpoint-dir", default=DEFAULT_CONFIG["checkpoint_dir"],
                   dest="checkpoint_dir")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Train ResNet-18 from scratch (not recommended).")
    p.add_argument("--gradcam", action="store_true",
                   help="Generate Grad-CAM visualisations after evaluation.")
    p.add_argument("--cross-val", action="store_true", dest="cross_val",
                   help="Run repeated-holdout cross-validation on clinical data (n=5 repeats).")
    p.add_argument("--monitor", default="recall",
                   choices=["accuracy", "recall", "f1", "auc"],
                   dest="monitor_metric",
                   help="Metric to monitor for early stopping (default: recall).")
    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()

    cfg = dict(DEFAULT_CONFIG)
    cfg.update({
        "public_dataset_path":   args.public,
        "clinical_dataset_path": args.clinical,
        "epochs":                args.epochs,
        "learning_rate":         args.learning_rate,
        "batch_size":            args.batch_size,
        "patience":              args.patience,
        "seed":                  args.seed,
        "num_workers":           args.num_workers,
        "checkpoint_dir":        args.checkpoint_dir,
    })

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_all_seeds(cfg["seed"])

    # Always define these so the cross-val block can reference them safely
    # regardless of which execution path (debug vs real data) was taken.
    clinical_ds: Optional[ClinicalDataset] = None
    public_ds:   Optional[PublicDataset] = None
    has_clinical = False
    has_public   = False

    print("\n" + "=" * 60)
    print("  Clinical AI — Oral Lesion Classification Pipeline")
    print("=" * 60)
    print(f"  Device : {device}")
    print(f"  Seed   : {cfg['seed']}")
    print(f"  Epochs : {cfg['epochs']}  (patience={cfg['patience']})")
    print(f"  LR     : {cfg['learning_rate']}   Batch: {cfg['batch_size']}")

    # ── DEBUG MODE ────────────────────────────────────────────────────────────
    if args.debug:
        print("\n[DEBUG MODE] Using synthetic data (60 samples, 3 epochs)\n")
        cfg["epochs"] = 3
        cfg["patience"] = 3
        loaders = build_debug_loaders(cfg, n_samples=60)
        mode = "debug"

    # ── REAL DATA MODE ────────────────────────────────────────────────────────
    else:
        print("\n[DATA LOADING]")
        clinical_ds = load_clinical_dataset(cfg["clinical_dataset_path"])
        public_ds   = load_public_dataset(cfg["public_dataset_path"])

        has_clinical = clinical_ds is not None
        has_public   = public_ds is not None

        if not has_clinical and not has_public:
            sys.exit(
                "\n[ERROR] No dataset found.\n"
                "  - Place images in data/public_dataset_1/images/ with a labels.csv\n"
                "  - Or run with --debug to use synthetic data\n"
                "  - Or pass --public <path> / --clinical <path>"
            )

        # Determine mode
        if args.mode == "clinical" or (args.mode is None and has_clinical):
            if not has_clinical:
                print("[WARN] --mode clinical requested but no clinical dataset found. "
                      "Falling back to public_only.")
                mode = "public_only"
            else:
                mode = "clinical"
        else:
            mode = "public_only"

        print(f"\n  Mode: {mode.upper()}")

        if mode == "public_only":
            if not has_public:
                sys.exit("[ERROR] public_only mode requires a public dataset.")
            loaders = build_public_only_loaders(public_ds, cfg)
            print(f"  Train: {len(loaders['train'].dataset)} | "
                  f"Val: {len(loaders['val'].dataset)} | "
                  f"Test: {len(loaders['test'].dataset)}")

        elif mode == "clinical":
            # Clinical dataset → val + test (NEVER trained on)
            # Public dataset   → train + val (pretraining / domain adaptation)
            #
            # This enforces the clean role separation:
            #   public  = pretraining signal (noisy labels)
            #   clinical = biopsy-confirmed ground truth for final evaluation only
            if not has_public:
                print("  [INFO] No public data — training entirely on clinical split.")
                loaders = build_clinical_loaders(clinical_ds, cfg)
            else:
                # Use public data for training; clinical for val/test only
                split_cfg = SplitConfig(random_seed=cfg["seed"])
                from dataset_split import split_clinical_dataset, split_public_dataset
                pub_splits = split_public_dataset(public_ds, split_cfg)
                clin_splits = split_clinical_dataset(clinical_ds, split_cfg)

                mean, std = _resolve_normalization_stats(public_ds, pub_splits["train"])
                tf = _make_transforms(mean, std)
                bs, nw = cfg["batch_size"], cfg["num_workers"]

                # Train on public train split
                # Val and test on held-out clinical splits
                loaders = {
                    "train": _subset_loader(public_ds, pub_splits["train"],
                                            tf["train"], bs, nw, shuffle=True),
                    "val":   _subset_loader(clinical_ds, clin_splits["val"],
                                            tf["val"], bs, nw),
                    "test":  _subset_loader(clinical_ds, clin_splits["test"],
                                            tf["test"], bs, nw),
                }
                print(f"  Train (public):    {len(loaders['train'].dataset)}")
                print(f"  Val   (clinical):  {len(loaders['val'].dataset)}")
                print(f"  Test  (clinical):  {len(loaders['test'].dataset)}")

    # ── BUILD MODEL ───────────────────────────────────────────────────────────
    print("\n[MODEL]")
    pretrained = not args.no_pretrained
    train_cfg = _make_training_config(cfg, args.monitor_metric)
    model = BinaryClassificationModel(
        model_type=cfg["model"],
        pretrained=pretrained,
        dropout_rate=cfg["dropout"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {cfg['model']}  |  trainable params: {n_params:,}  |  pretrained={pretrained}")
    print(f"  Early-stopping metric: {train_cfg.monitor_metric}")

    # ── TRAIN ─────────────────────────────────────────────────────────────────
    print(f"\n[TRAINING — {mode.upper()}]")
    if mode == "clinical" and has_public:
        # Phase A: domain pre-training on public data, validated on clinical held-out val.
        # The two-phase approach is the scientifically correct setup: the model never
        # sees clinical training labels, only clinical validation for early stopping.
        phase_a_domain_training(model, loaders["train"], loaders["val"], train_cfg)
        # Reload the best-on-val checkpoint (not the final-epoch checkpoint)
        best_a_ckpt = str(Path(train_cfg.checkpoint_dir) / "phase_a_best.pth")
        load_checkpoint(model, best_a_ckpt, device)
    else:
        # Single-phase: debug / public_only / clinical without public pretraining data
        model = _train_single_phase(model, loaders, train_cfg, device)

    # ── EVALUATE ──────────────────────────────────────────────────────────────
    print("\n[EVALUATION]")
    eval_cfg = EvaluationConfig(output_dir="evaluation_results", device=device)
    _full_evaluate(model, loaders["test"], eval_cfg)
    print("=" * 60 + "\n")

    # ── GRAD-CAM ──────────────────────────────────────────────────────────────
    if args.gradcam:
        print("\n[GRAD-CAM]")
        try:
            from gradcam import get_target_layer, save_gradcam_examples
            target_layer = get_target_layer(model, model_type=cfg["model"])
            save_gradcam_examples(
                model=model,
                dataloader=loaders["test"],
                output_dir="evaluation_results/gradcam",
                target_layer=target_layer,
                device=device,
            )
        except Exception as e:
            print(f"  [WARN] Grad-CAM failed: {e}")

    # ── CROSS-VALIDATION ──────────────────────────────────────────────────────
    if args.cross_val:
        if mode == "debug":
            print("\n[CV] Cross-validation skipped in debug mode.")
        else:
            # clinical_ds and public_ds are always defined (possibly None) because
            # they are initialised at the top of main() before any branching.
            _cv_dataset = clinical_ds if mode == "clinical" else public_ds

            if _cv_dataset is not None:
                print("\n[CROSS-VALIDATION]  5 × repeated holdout")
                from cross_validation import run_repeated_holdout, save_cv_report

                def _train_eval_fn(cv_loaders: Dict[str, DataLoader]) -> Dict:
                    cv_model = BinaryClassificationModel(
                        model_type=cfg["model"],
                        pretrained=(not args.no_pretrained),
                        dropout_rate=cfg["dropout"],
                    ).to(device)
                    cv_model = _train_single_phase(
                        cv_model, cv_loaders, train_cfg, device
                    )
                    criterion = nn.CrossEntropyLoss()
                    m = validate_epoch(cv_model, cv_loaders["test"], criterion, device)
                    # Map to keys expected by cross_validation._summarise
                    return {
                        "accuracy":    m.get("accuracy"),
                        "sensitivity": m.get("recall"),  # recall == sensitivity
                        "f1":          m.get("f1"),
                        "auc":         m.get("auc"),
                    }

                cv_results = run_repeated_holdout(
                    dataset=_cv_dataset,
                    train_eval_fn=_train_eval_fn,
                    n_repeats=5,
                    batch_size=cfg["batch_size"],
                    num_workers=cfg["num_workers"],
                )
                save_cv_report(
                    cv_results, output_path="evaluation_results/cv_results"
                )
            else:
                print("\n[CV] No dataset available for cross-validation.")


if __name__ == "__main__":
    main()
