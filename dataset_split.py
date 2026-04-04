"""
Dataset Splitting and DataLoader Creation Module

This module handles:
- Stratified train/val/test splitting for clinical data
- Train/val splitting for public data
- Prevention of data leakage
- Creation of PyTorch DataLoaders with proper transforms

Scientific requirements:
- Stratification on biopsy diagnoses for clinical data
- Test set is strictly held out (never used during development)
- No cross-contamination between splits
- Reproducible splits with seed control
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from data_loader import BaseDataset, ClinicalDataset, PublicDataset
from preprocessing import (
    TransformPipeline,
    AugmentationConfig,
    TransformedDataset,
)


logger = logging.getLogger(__name__)


# ============================================================================
# SPLIT CONFIGURATION
# ============================================================================

class SplitConfig:
    """
    Configuration for dataset splitting.

    This class is the single source of truth for split ratios used throughout
    the repository. Avoid duplicating train/val/test ratios in other modules.
    
    Attributes:
        random_seed: Random seed for reproducibility
        clinical_train_ratio: Ratio for clinical train set (0.0-1.0)
        clinical_val_ratio: Ratio for clinical val set (0.0-1.0)
        clinical_test_ratio: Ratio for clinical test set (0.0-1.0)
        public_train_ratio: Ratio for public train set (0.0-1.0)
        public_val_ratio: Ratio for public val set (0.0-1.0)
        stratify_on_label: Whether to use stratified splitting
    """
    
    # Clinical dataset split (must sum to 1.0)
    CLINICAL_TRAIN_RATIO = 0.70
    CLINICAL_VAL_RATIO = 0.15
    CLINICAL_TEST_RATIO = 0.15
    
    # Public dataset split (must sum to 1.0)
    PUBLIC_TRAIN_RATIO = 0.80
    PUBLIC_VAL_RATIO = 0.20
    
    # Stratification
    STRATIFY_ON_LABEL = True

    # Patient-level grouping (prevents multi-image patient leakage)
    PATIENT_LEVEL_SPLIT = True
    
    # Reproducibility
    RANDOM_SEED = 42
    
    def __init__(
        self,
        clinical_train_ratio: float = CLINICAL_TRAIN_RATIO,
        clinical_val_ratio: float = CLINICAL_VAL_RATIO,
        clinical_test_ratio: float = CLINICAL_TEST_RATIO,
        public_train_ratio: float = PUBLIC_TRAIN_RATIO,
        public_val_ratio: float = PUBLIC_VAL_RATIO,
        stratify_on_label: bool = STRATIFY_ON_LABEL,
        patient_level_split: bool = PATIENT_LEVEL_SPLIT,
        random_seed: int = RANDOM_SEED,
    ):
        """Initialize split configuration."""
        self.clinical_train_ratio = clinical_train_ratio
        self.clinical_val_ratio = clinical_val_ratio
        self.clinical_test_ratio = clinical_test_ratio
        self.public_train_ratio = public_train_ratio
        self.public_val_ratio = public_val_ratio
        self.stratify_on_label = stratify_on_label
        self.patient_level_split = patient_level_split
        self.random_seed = random_seed
        
        # Validate splits sum to 1.0
        clinical_sum = clinical_train_ratio + clinical_val_ratio + clinical_test_ratio
        if not np.isclose(clinical_sum, 1.0, atol=0.01):
            raise ValueError(
                f"Clinical split ratios must sum to 1.0, got {clinical_sum}"
            )
        
        public_sum = public_train_ratio + public_val_ratio
        if not np.isclose(public_sum, 1.0, atol=0.01):
            raise ValueError(
                f"Public split ratios must sum to 1.0, got {public_sum}"
            )
        
        logger.info(
            f"Split config: "
            f"clinical={clinical_train_ratio:.0%}/{clinical_val_ratio:.0%}/{clinical_test_ratio:.0%}, "
            f"public={public_train_ratio:.0%}/{public_val_ratio:.0%}, "
            f"seed={random_seed}"
        )


# ============================================================================
# CLINICAL DATASET SPLITTING — helpers
# ============================================================================

def _split_by_patient(
    dataset: ClinicalDataset,
    indices: np.ndarray,
    labels: np.ndarray,
    patient_ids: List[Optional[str]],
    split_config: SplitConfig,
) -> Dict[str, List[int]]:
    """
    Patient-level stratified splitting.

    Groups all image indices belonging to the same patient, then splits at the
    patient level so that every image from a given patient always lands in the
    same split.  Stratification uses the label of the first image seen for each
    patient (biopsy labels are per-patient in practice).

    Parameters
    ----------
    dataset     : ClinicalDataset whose samples are being split
    indices     : full index array (0 … N-1)
    labels      : label array aligned with indices
    patient_ids : patient_id string for each sample (no Nones)
    split_config: split ratios and seed

    Returns
    -------
    dict with keys 'train', 'val', 'test' containing image-level indices
    """
    patient_to_indices: Dict[str, List[int]] = defaultdict(list)
    patient_to_label: Dict[str, int] = {}

    for idx, pid in zip(indices.tolist(), patient_ids):
        patient_to_indices[pid].append(idx)
        if pid not in patient_to_label:
            patient_to_label[pid] = int(labels[idx])

    unique_patients = list(patient_to_indices.keys())
    patient_labels = np.array([patient_to_label[p] for p in unique_patients])

    logger.info(
        f"  Patient-level split: {len(unique_patients)} unique patients "
        f"(label distribution: {np.bincount(patient_labels)})"
    )

    # Step 1: carve out test patients
    stratify = patient_labels if split_config.stratify_on_label else None
    patients_trainval, patients_test = train_test_split(
        unique_patients,
        test_size=split_config.clinical_test_ratio,
        stratify=stratify,
        random_state=split_config.random_seed,
    )

    # Step 2: split remaining into train / val
    trainval_labels = np.array([patient_to_label[p] for p in patients_trainval])
    train_ratio = split_config.clinical_train_ratio / (
        split_config.clinical_train_ratio + split_config.clinical_val_ratio
    )
    stratify_tv = trainval_labels if split_config.stratify_on_label else None
    patients_train, patients_val = train_test_split(
        patients_trainval,
        train_size=train_ratio,
        stratify=stratify_tv,
        random_state=split_config.random_seed,
    )

    # Assert patients never appear in multiple splits
    set_train = set(patients_train)
    set_val   = set(patients_val)
    set_test  = set(patients_test)
    assert not (set_train & set_val),  "Patient leakage: same patient in train AND val!"
    assert not (set_train & set_test), "Patient leakage: same patient in train AND test!"
    assert not (set_val   & set_test), "Patient leakage: same patient in val AND test!"

    # Expand patient lists → image-index lists
    train_indices = [i for p in patients_train for i in patient_to_indices[p]]
    val_indices   = [i for p in patients_val   for i in patient_to_indices[p]]
    test_indices  = [i for p in patients_test  for i in patient_to_indices[p]]

    logger.info(
        f"  Train: {len(patients_train)} patients, {len(train_indices)} images, "
        f"labels={np.bincount(labels[train_indices])}"
    )
    logger.info(
        f"  Val:   {len(patients_val)} patients, {len(val_indices)} images, "
        f"labels={np.bincount(labels[val_indices])}"
    )
    logger.info(
        f"  Test:  {len(patients_test)} patients, {len(test_indices)} images, "
        f"labels={np.bincount(labels[test_indices])}"
    )

    return {
        'train': train_indices,
        'val':   val_indices,
        'test':  test_indices,
    }


def _split_by_image(
    indices: np.ndarray,
    labels: np.ndarray,
    split_config: SplitConfig,
) -> Dict[str, List[int]]:
    """
    Image-level stratified splitting (fallback when patient_id is unavailable).

    Parameters
    ----------
    indices     : full index array (0 … N-1)
    labels      : label array aligned with indices
    split_config: split ratios and seed

    Returns
    -------
    dict with keys 'train', 'val', 'test' containing image-level indices
    """
    stratify = labels if split_config.stratify_on_label else None

    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=split_config.clinical_test_ratio,
        stratify=stratify,
        random_state=split_config.random_seed,
    )

    train_val_labels = labels[train_val_indices]
    train_size_ratio = split_config.clinical_train_ratio / (
        split_config.clinical_train_ratio + split_config.clinical_val_ratio
    )
    stratify_tv = train_val_labels if split_config.stratify_on_label else None

    train_indices, val_indices = train_test_split(
        train_val_indices,
        train_size=train_size_ratio,
        stratify=stratify_tv,
        random_state=split_config.random_seed,
    )

    assert len(set(train_indices) & set(val_indices))  == 0, "Train/val overlap!"
    assert len(set(train_indices) & set(test_indices)) == 0, "Train/test overlap!"
    assert len(set(val_indices)   & set(test_indices)) == 0, "Val/test overlap!"

    logger.info(
        f"  Train: n={len(train_indices)}, labels={np.bincount(labels[train_indices])}"
    )
    logger.info(
        f"  Val:   n={len(val_indices)}, labels={np.bincount(labels[val_indices])}"
    )
    logger.info(
        f"  Test:  n={len(test_indices)}, labels={np.bincount(labels[test_indices])}"
    )

    return {
        'train': train_indices.tolist(),
        'val':   val_indices.tolist(),
        'test':  test_indices.tolist(),
    }


# ============================================================================
# CLINICAL DATASET SPLITTING
# ============================================================================

def split_clinical_dataset(
    dataset: ClinicalDataset,
    split_config: SplitConfig = None,
) -> Dict[str, List[int]]:
    """
    Split clinical dataset into train/val/test with stratification.

    When ``split_config.patient_level_split`` is True (default) and all samples
    carry a non-None ``patient_id``, the split happens at the patient level:
    every image from the same patient is guaranteed to land in the same subset,
    preventing data leakage caused by multi-image patients.

    If any ``patient_id`` is None a warning is emitted and the function falls
    back to image-level splitting.

    An assertion verifies that no patient_id appears in more than one split.

    Parameters
    ----------
    dataset      : ClinicalDataset instance
    split_config : SplitConfig instance (uses defaults if None)

    Returns
    -------
    dict with keys 'train', 'val', 'test' containing sample indices
    """
    if split_config is None:
        split_config = SplitConfig()

    labels  = np.array([metadata.label for _, metadata in dataset.samples])
    indices = np.arange(len(dataset))

    logger.info(f"Splitting clinical dataset: n={len(dataset)}")
    logger.info(f"  Label distribution: {np.bincount(labels)}")

    # ------------------------------------------------------------------
    # Patient-level splitting (preferred)
    # ------------------------------------------------------------------
    if split_config.patient_level_split:
        patient_ids = [metadata.patient_id for _, metadata in dataset.samples]
        n_missing = sum(1 for pid in patient_ids if pid is None)

        if n_missing == 0:
            logger.info("  Using patient-level splitting (no leakage for multi-image patients)")
            return _split_by_patient(dataset, indices, labels, patient_ids, split_config)
        else:
            logger.warning(
                f"patient_level_split=True but {n_missing}/{len(patient_ids)} samples "
                "have patient_id=None. Falling back to image-level splitting. "
                "Add a 'patient_id' column to the metadata CSV to enable patient-level grouping."
            )

    # ------------------------------------------------------------------
    # Image-level splitting (fallback)
    # ------------------------------------------------------------------
    logger.info("  Using image-level splitting")
    return _split_by_image(indices, labels, split_config)


# ============================================================================
# PUBLIC DATASET SPLITTING
# ============================================================================

def split_public_dataset(
    dataset: PublicDataset,
    split_config: SplitConfig = None,
) -> Dict[str, List[int]]:
    """
    Split public dataset into train/val.
    
    Uses stratified splitting if available to maintain label distribution.
    
    Args:
        dataset: PublicDataset instance
        split_config: SplitConfig instance
        
    Returns:
        Dictionary with keys 'train', 'val' containing indices
    """
    if split_config is None:
        split_config = SplitConfig()
    
    # Get labels for stratification
    labels = np.array([metadata.label for _, metadata in dataset.samples])
    indices = np.arange(len(dataset))
    
    # FIX (global state): np.random.seed() removed. train_test_split's random_state=
    # parameter guarantees reproducibility without mutating global random state.
    
    logger.info(f"Splitting public dataset: n={len(dataset)}")
    logger.info(f"  Label distribution: {np.bincount(labels)}")
    
    # Split train and validation
    val_size = split_config.public_val_ratio
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_size,
        stratify=labels if split_config.stratify_on_label else None,
        random_state=split_config.random_seed,
    )
    
    # Verify no overlap
    assert len(set(train_indices) & set(val_indices)) == 0, "Train/val overlap!"
    
    # Log split statistics
    train_labels = labels[train_indices]
    val_labels = labels[val_indices]
    
    logger.info(f"Public split results:")
    logger.info(f"  Train: n={len(train_indices)}, labels={np.bincount(train_labels)}")
    logger.info(f"  Val:   n={len(val_indices)}, labels={np.bincount(val_labels)}")
    
    return {
        'train': train_indices.tolist(),
        'val': val_indices.tolist(),
    }


# ============================================================================
# DATALOADER CREATION
# ============================================================================

class DataLoaderFactory:
    """
    Factory for creating train/val/test DataLoaders with proper transforms.
    
    Handles:
    - Application of transforms to datasets
    - Stratified splitting
    - DataLoader creation with proper batch size and workers
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        batch_size: int = 32,
        num_workers: int = 4,
        augmentation_config: Optional[AugmentationConfig] = None,
        split_config: Optional[SplitConfig] = None,
    ):
        """
        Initialize DataLoader factory.
        
        Args:
            image_size: Target image size
            batch_size: Batch size for DataLoaders
            num_workers: Number of worker processes
            augmentation_config: Augmentation configuration
            split_config: Split configuration
        """
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.augmentation_config = augmentation_config or AugmentationConfig()
        self.split_config = split_config or SplitConfig()
        
        # Create transform pipeline
        self.transform_pipeline = TransformPipeline(
            image_size=image_size,
            augmentation_config=self.augmentation_config,
        )
        
        self.transforms = self.transform_pipeline.get_transforms_dict()
    
    def create_clinical_dataloaders(
        self,
        clinical_dataset: ClinicalDataset,
    ) -> Dict[str, DataLoader]:
        """
        Create train/val/test DataLoaders for clinical dataset.
        
        Args:
            clinical_dataset: ClinicalDataset instance
            
        Returns:
            Dictionary with keys 'train', 'val', 'test'
        """
        # Split dataset
        splits = split_clinical_dataset(
            clinical_dataset,
            self.split_config
        )
        
        # Create transformed subsets
        train_subset = Subset(clinical_dataset, splits['train'])
        val_subset = Subset(clinical_dataset, splits['val'])
        test_subset = Subset(clinical_dataset, splits['test'])
        
        # Apply transforms
        train_transformed = TransformedDataset(train_subset, self.transforms['train'])
        val_transformed = TransformedDataset(val_subset, self.transforms['val'])
        test_transformed = TransformedDataset(test_subset, self.transforms['test'])
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_transformed,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        
        val_loader = DataLoader(
            val_transformed,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        
        test_loader = DataLoader(
            test_transformed,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        
        logger.info(
            f"Created clinical dataloaders: "
            f"train={len(train_loader)} batches, "
            f"val={len(val_loader)} batches, "
            f"test={len(test_loader)} batches"
        )
        
        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader,
        }
    
    def create_public_dataloaders(
        self,
        public_dataset: PublicDataset,
    ) -> Dict[str, DataLoader]:
        """
        Create train/val DataLoaders for public dataset.
        
        Args:
            public_dataset: PublicDataset instance
            
        Returns:
            Dictionary with keys 'train', 'val'
        """
        # Split dataset
        splits = split_public_dataset(
            public_dataset,
            self.split_config
        )
        
        # Create transformed subsets
        train_subset = Subset(public_dataset, splits['train'])
        val_subset = Subset(public_dataset, splits['val'])
        
        # Apply transforms
        train_transformed = TransformedDataset(train_subset, self.transforms['train'])
        val_transformed = TransformedDataset(val_subset, self.transforms['val'])
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_transformed,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        
        val_loader = DataLoader(
            val_transformed,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        
        logger.info(
            f"Created public dataloaders: "
            f"train={len(train_loader)} batches, "
            f"val={len(val_loader)} batches"
        )
        
        return {
            'train': train_loader,
            'val': val_loader,
        }
    
    def create_combined_dataloaders(
        self,
        clinical_dataset: Optional[ClinicalDataset] = None,
        public_datasets: Optional[List[PublicDataset]] = None,
    ) -> Dict[str, Dict[str, DataLoader]]:
        """
        Create DataLoaders for multiple datasets.
        
        Args:
            clinical_dataset: Clinical dataset (optional)
            public_datasets: List of public datasets (optional)
            
        Returns:
            Dictionary mapping dataset names to loaders
        """
        dataloaders = {}
        
        if clinical_dataset is not None:
            dataloaders['clinical'] = self.create_clinical_dataloaders(
                clinical_dataset
            )
        
        if public_datasets is not None:
            for i, public_ds in enumerate(public_datasets):
                dataset_key = f'public_{i+1}'
                dataloaders[dataset_key] = self.create_public_dataloaders(
                    public_ds
                )
        
        return dataloaders


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_clinical_dataloaders(
    clinical_dataset: ClinicalDataset,
    batch_size: int = 32,
    num_workers: int = 4,
    augmentation_config: Optional[AugmentationConfig] = None,
    split_config: Optional[SplitConfig] = None,
) -> Dict[str, DataLoader]:
    """
    Convenience function to create clinical dataloaders.
    
    Args:
        clinical_dataset: ClinicalDataset instance
        batch_size: Batch size
        num_workers: Number of workers
        augmentation_config: Augmentation configuration
        split_config: Split configuration
        
    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders
    """
    factory = DataLoaderFactory(
        batch_size=batch_size,
        num_workers=num_workers,
        augmentation_config=augmentation_config,
        split_config=split_config,
    )
    
    return factory.create_clinical_dataloaders(clinical_dataset)


def get_public_dataloaders(
    public_dataset: PublicDataset,
    batch_size: int = 32,
    num_workers: int = 4,
    augmentation_config: Optional[AugmentationConfig] = None,
    split_config: Optional[SplitConfig] = None,
) -> Dict[str, DataLoader]:
    """
    Convenience function to create public dataloaders.
    
    Args:
        public_dataset: PublicDataset instance
        batch_size: Batch size
        num_workers: Number of workers
        augmentation_config: Augmentation configuration
        split_config: Split configuration
        
    Returns:
        Dictionary with 'train', 'val' DataLoaders
    """
    factory = DataLoaderFactory(
        batch_size=batch_size,
        num_workers=num_workers,
        augmentation_config=augmentation_config,
        split_config=split_config,
    )
    
    return factory.create_public_dataloaders(public_dataset)


def get_all_dataloaders(
    clinical_dataset: Optional[ClinicalDataset] = None,
    public_datasets: Optional[List[PublicDataset]] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    augmentation_config: Optional[AugmentationConfig] = None,
    split_config: Optional[SplitConfig] = None,
) -> Dict[str, Dict[str, DataLoader]]:
    """
    Convenience function to create all dataloaders.
    
    Args:
        clinical_dataset: Clinical dataset (optional)
        public_datasets: List of public datasets (optional)
        batch_size: Batch size
        num_workers: Number of workers
        augmentation_config: Augmentation configuration
        split_config: Split configuration
        
    Returns:
        Nested dictionary: {dataset_name: {split: DataLoader}}
    """
    factory = DataLoaderFactory(
        batch_size=batch_size,
        num_workers=num_workers,
        augmentation_config=augmentation_config,
        split_config=split_config,
    )
    
    return factory.create_combined_dataloaders(
        clinical_dataset=clinical_dataset,
        public_datasets=public_datasets,
    )


# ============================================================================
# VALIDATION AND REPORTING
# ============================================================================

def report_split_statistics(
    dataloaders: Dict[str, DataLoader],
    dataset_name: str = "Dataset",
) -> Dict[str, Any]:
    """
    Report statistics about a split.
    
    Args:
        dataloaders: Dictionary of DataLoaders
        dataset_name: Name of dataset for logging
        
    Returns:
        Dictionary with statistics
    """
    stats = {}
    
    for split_name, loader in dataloaders.items():
        all_labels = []
        num_batches = len(loader)
        
        for _, labels, _ in loader:
            all_labels.extend(labels.cpu().numpy().tolist())
        
        all_labels = np.array(all_labels)
        
        label_counts = np.bincount(all_labels.astype(int))
        if len(label_counts) == 1:
            # Only one class
            label_counts = np.append(label_counts, 0) if label_counts[0] > 0 \
                else np.prepend(0, label_counts)
        
        stats[split_name] = {
            'num_samples': len(all_labels),
            'num_batches': num_batches,
            'label_0_count': label_counts[0] if len(label_counts) > 0 else 0,
            'label_1_count': label_counts[1] if len(label_counts) > 1 else 0,
            'label_0_percent': 100 * label_counts[0] / len(all_labels) \
                if len(all_labels) > 0 else 0,
            'label_1_percent': 100 * label_counts[1] / len(all_labels) \
                if len(all_labels) > 0 else 0,
        }
    
    # Log statistics
    logger.info(f"\n{'='*60}")
    logger.info(f"Split Statistics: {dataset_name}")
    logger.info(f"{'='*60}")
    
    for split_name, split_stats in stats.items():
        logger.info(f"\n{split_name.upper()}:")
        logger.info(f"  Samples: {split_stats['num_samples']}")
        logger.info(f"  Batches: {split_stats['num_batches']}")
        logger.info(
            f"  Labels: {split_stats['label_0_count']} (0) / "
            f"{split_stats['label_1_count']} (1)"
        )
        logger.info(
            f"  Distribution: {split_stats['label_0_percent']:.1f}% / "
            f"{split_stats['label_1_percent']:.1f}%"
        )
    
    logger.info(f"{'='*60}\n")
    
    return stats


if __name__ == "__main__":
    """
    Example usage of dataset splitting and dataloader creation.
    """
    print("Dataset splitting and dataloader creation module")
    print("\nUsage:")
    print("  from dataset_split import get_clinical_dataloaders, SplitConfig")
    print("  from data_loader import get_clinical_dataset")
    print("  from preprocessing import AugmentationConfig")
    print()
    print("  clinical_dataset = get_clinical_dataset('data/clinical_dataset')")
    print("  dataloaders = get_clinical_dataloaders(")
    print("      clinical_dataset,")
    print("      batch_size=32,")
    print("      num_workers=4,")
    print("  )")
    print()
    print("  for images, labels, metadata in dataloaders['train']:")
    print("      # Training code")
    print("      pass")
