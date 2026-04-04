"""
Configuration file for data loading and model training.

This module centralizes configuration for dataset paths, data loading parameters,
and model hyperparameters. Import DataLoadingConfig and DATASETS_CONFIG from
here rather than duplicating constants across modules.

Dataset split ratios are intentionally not defined in this module. The single
source of truth for split ratios is dataset_split.SplitConfig.
"""

import json
from pathlib import Path
from typing import Dict, Any

import yaml


# ============================================================================
# DATA PATHS
# ============================================================================

# Base data directory
DATA_DIR = Path(__file__).parent / "data"

# Dataset paths
PUBLIC_DATASET_1_PATH = DATA_DIR / "public_dataset_1"
PUBLIC_DATASET_2_PATH = DATA_DIR / "public_dataset_2"
CLINICAL_DATASET_PATH = DATA_DIR / "clinical_dataset"

# Dataset configurations
DATASETS_CONFIG = {
    'public_1': {
        'path': PUBLIC_DATASET_1_PATH,
        'labels_file': 'labels.csv',
        'image_dir': 'images',
        'label_column': 'label',
        'image_name_column': 'image_name',
        'type': 'public',
    },
    'public_2': {
        'path': PUBLIC_DATASET_2_PATH,
        'labels_file': 'labels.csv',
        'image_dir': 'images',
        'label_column': 'label',
        'image_name_column': 'image_name',
        'type': 'public',
    },
    'clinical': {
        'path': CLINICAL_DATASET_PATH,
        'metadata_file': 'metadata.csv',
        'image_dir': 'images',
        'id_column': 'id',
        'image_name_column': 'image_name',
        'label_column': 'biopsy_diagnosis',
        'clinician_diagnosis_column': 'clinician_diagnosis',
        'lesion_type_column': 'lesion_type',
        'location_column': 'location',
        'type': 'clinical',
    },
}


# ============================================================================
# DATA LOADING PARAMETERS
# ============================================================================

class DataLoadingConfig:
    """Configuration for data loading."""
    
    # Image preprocessing
    IMAGE_SIZE = (224, 224)
    NORMALIZE = True
    NORMALIZATION_MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization
    NORMALIZATION_STD = [0.229, 0.224, 0.225]
    
    # DataLoader parameters
    BATCH_SIZE = 32
    TEST_BATCH_SIZE = 64
    NUM_WORKERS = 4
    PIN_MEMORY = True
    SHUFFLE_TRAIN = True
    SHUFFLE_VAL = False
    SHUFFLE_TEST = False
    
    # Data augmentation (optional)
    USE_AUGMENTATION = True
    AUGMENTATION_CONFIG = {
        'random_horizontal_flip': 0.5,
        'random_vertical_flip': 0.5,
        'random_rotation': 15,  # degrees
        'random_brightness': 0.2,
        'random_contrast': 0.2,
    }


# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

class ModelConfig:
    """Configuration for model training."""
    
    # Model architecture
    MODEL_NAME = 'resnet50'
    NUM_CLASSES = 2
    PRETRAINED = True
    DROPOUT_RATE = 0.5
    
    # Training parameters
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    OPTIMIZER = 'adam'
    SCHEDULER = 'cosine'  # 'cosine', 'linear', 'exponential'
    
    # Loss function
    LOSS_FUNCTION = 'cross_entropy'
    CLASS_WEIGHTS = None  # Set to [weight_0, weight_1] for imbalanced data
    
    # Checkpointing
    CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
    SAVE_FREQUENCY = 5  # Save every N epochs
    EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 10


# ============================================================================
# EXPERIMENT TRACKING
# ============================================================================

class PreprocessingConfig:
    """Configuration for image preprocessing and augmentation."""
    
    # Image settings
    IMAGE_SIZE = (224, 224)
    
    # Normalization (ImageNet by default)
    NORMALIZATION_MEAN = DataLoadingConfig.NORMALIZATION_MEAN
    NORMALIZATION_STD = DataLoadingConfig.NORMALIZATION_STD
    
    # Augmentation (for training only)
    USE_AUGMENTATION = True
    
    # Augmentation parameters (conservative, clinically appropriate)
    RANDOM_HORIZONTAL_FLIP_PROB = 0.5
    RANDOM_VERTICAL_FLIP_PROB = 0.3
    RANDOM_ROTATION_DEGREES = 15
    BRIGHTNESS_FACTOR = 0.1
    CONTRAST_FACTOR = 0.1
    SATURATION_FACTOR = 0.1
    HUE_FACTOR = 0.05
    AFFINE_TRANSLATE_PERCENT = (0.05, 0.05)
    SHEAR_DEGREES = (-5, 5)
    PERSPECTIVE_DISTORTION_SCALE = 0.1


# Deprecated: split ratios now live exclusively in dataset_split.SplitConfig.
# This compatibility shim remains only to avoid a hard runtime failure in any
# external scripts that still import SplittingConfig from config.py.
class SplittingConfig:
    """Deprecated compatibility shim. Use dataset_split.SplitConfig instead."""

    CLINICAL_TRAIN_RATIO = 0.70
    CLINICAL_VAL_RATIO = 0.15
    CLINICAL_TEST_RATIO = 0.15
    PUBLIC_TRAIN_RATIO = 0.80
    PUBLIC_VAL_RATIO = 0.20
    STRATIFY_ON_LABEL = True
    RANDOM_SEED = 42


class ExperimentConfig:
    """Configuration for experiment tracking and logging."""
    
    # Logging
    LOG_DIR = Path(__file__).parent / "logs"
    LOG_LEVEL = "INFO"
    LOG_TO_FILE = True
    
    # Metrics tracking
    TRACK_METRICS = [
        'accuracy',
        'precision',
        'recall',
        'f1',
        'auroc',
    ]
    
    # Wandb (optional)
    USE_WANDB = False
    WANDB_PROJECT = "clinical-ai-oral-lesion"
    WANDB_ENTITY = None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'clinical', 'public_1')
        
    Returns:
        Dictionary containing dataset configuration
        
    Raises:
        KeyError: If dataset name not found
    """
    if dataset_name not in DATASETS_CONFIG:
        raise KeyError(
            f"Unknown dataset: {dataset_name}. "
            f"Available datasets: {list(DATASETS_CONFIG.keys())}"
        )
    
    return DATASETS_CONFIG[dataset_name]


def load_config_from_file(config_path: Path) -> Dict[str, Any]:
    """
    Load configuration from a YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Dictionary containing configuration
        
    Raises:
        FileNotFoundError: If config file not found
        ValueError: If unsupported file format
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    else:
        raise ValueError(
            f"Unsupported config format: {config_path.suffix}. "
            f"Use .yaml or .json"
        )
    
    return config


def save_config_to_file(config: Dict[str, Any], config_path: Path) -> None:
    """
    Save configuration to a YAML or JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    elif config_path.suffix == '.json':
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    else:
        raise ValueError(
            f"Unsupported config format: {config_path.suffix}. "
            f"Use .yaml or .json"
        )


if __name__ == "__main__":
    """Print example configuration."""
    print("Data Loading Config:")
    print(f"  Image size: {DataLoadingConfig.IMAGE_SIZE}")
    print(f"  Batch size: {DataLoadingConfig.BATCH_SIZE}")
    print(f"  Num workers: {DataLoadingConfig.NUM_WORKERS}")
    
    print("\nModel Config:")
    print(f"  Model: {ModelConfig.MODEL_NAME}")
    print(f"  Num epochs: {ModelConfig.NUM_EPOCHS}")
    print(f"  Learning rate: {ModelConfig.LEARNING_RATE}")
    
    print("\nAvailable datasets:")
    for name in DATASETS_CONFIG.keys():
        print(f"  - {name}")
