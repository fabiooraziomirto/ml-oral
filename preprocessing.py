"""
Image Preprocessing and Data Augmentation Module

This module provides image transformation pipelines for training, validation,
and testing. Includes data augmentation strategies specifically designed for
medical imaging with controlled, clinically-relevant augmentations.

Features:
- ImageNet normalization as a safe fallback
- Optional dataset-specific normalization computed from training data
- Data augmentation for training (controlled and clinically relevant)
- Separate pipelines for train/val/test
- Configurable augmentation parameters
- Reproducible random state management

# Augmentation design rationale

Clinical image augmentation must preserve lesion morphology, chromatic cues, and
surrounding tissue context. The strategy in this module is intentionally
conservative: it favours small acquisition-like perturbations over aggressive
image synthesis. Specifically, the pipeline excludes elastic deformation,
aggressive hue/saturation jitter, random erasing/cutout, and extreme zoom
because those operations can alter diagnostic borders, colour, or context.
"""

from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import logging

from config import DataLoadingConfig
from PIL import Image
import torch
from torchvision import transforms
import numpy as np


logger = logging.getLogger(__name__)


# ============================================================================
# AUGMENTATION CONFIGURATION
# ============================================================================

@dataclass
class AugmentationConfig:
    """
    Configuration for data augmentation parameters.
    
    All augmentations are clinically-motivated and carefully tuned to preserve
    diagnostic information while providing sufficient variation for model robustness.
    """
    
    # Random Flip
    random_horizontal_flip_prob: float = 0.5  # Natural variation
    random_vertical_flip_prob: float = 0.3    # Natural variation (less common)
    
    # Rotation
    random_rotation_degrees: int = 15  # ±15 degrees (clinically valid)
    
    # Color jitter is intentionally light. Aggressive hue (> 0.1) or saturation
    # (> 0.3) shifts would distort clinically meaningful mucosal colour signals.
    brightness_factor: float = 0.1    # Lighting variation
    contrast_factor: float = 0.1      # Contrast variation
    saturation_factor: float = 0.1    # Color saturation
    hue_factor: float = 0.05          # Hue variation (subtle)
    
    # Random Affine (slight displacement)
    affine_translate_percent: Tuple[float, float] = (0.05, 0.05)  # 5% translation
    shear_degrees: Tuple[float, float] = (-5, 5)  # Subtle shearing
    
    # Random Perspective (mild)
    perspective_distortion_scale: float = 0.1
    
    # Elastic deformation is intentionally disabled because it can change lesion
    # borders and create anatomically implausible morphology.
    use_elastic_deformation: bool = False
    
    def __post_init__(self):
        """Validate augmentation parameters."""
        if not (0 <= self.random_horizontal_flip_prob <= 1):
            raise ValueError("random_horizontal_flip_prob must be in [0, 1]")
        if not (0 <= self.random_vertical_flip_prob <= 1):
            raise ValueError("random_vertical_flip_prob must be in [0, 1]")
        if self.random_rotation_degrees < 0:
            raise ValueError("random_rotation_degrees must be non-negative")
        logger.info(f"Augmentation config validated: "
                   f"flip_h={self.random_horizontal_flip_prob}, "
                   f"flip_v={self.random_vertical_flip_prob}, "
                   f"rotation={self.random_rotation_degrees}°")


# ============================================================================
# NORMALIZATION STATISTICS
# ============================================================================

# ImageNet normalization — sourced from config.py to avoid duplication
IMAGENET_MEAN = DataLoadingConfig.NORMALIZATION_MEAN
IMAGENET_STD  = DataLoadingConfig.NORMALIZATION_STD

# Medical imaging normalization (alternative — use if fine-tuning on medical data)
MEDICAL_MEAN = DataLoadingConfig.NORMALIZATION_MEAN  # adjust with dataset-specific stats
MEDICAL_STD  = DataLoadingConfig.NORMALIZATION_STD


def compute_dataset_stats(dataset, sample_size: int = 500) -> Tuple[List[float], List[float]]:
    """
    Estimate per-channel mean and std from up to ``sample_size`` images.

    Images are resized to the configured network input resolution and converted
    to tensors in [0, 1] before accumulation.

    Parameters
    ----------
    dataset     : dataset-like object with ``len`` and ``__getitem__``
    sample_size : maximum number of images to sample

    Returns
    -------
    tuple of ([mean_r, mean_g, mean_b], [std_r, std_g, std_b])
    """
    n_items = len(dataset)
    if n_items == 0:
        raise ValueError("Cannot compute dataset statistics on an empty dataset")

    rng = np.random.default_rng(42)
    n_samples = min(int(sample_size), n_items)
    sampled_indices = rng.choice(n_items, size=n_samples, replace=False)
    to_tensor = transforms.Compose([
        transforms.Resize(DataLoadingConfig.IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sq_sum = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0

    for idx in sampled_indices.tolist():
        image = dataset[idx][0]
        if isinstance(image, Image.Image):
            tensor = to_tensor(image)
        elif isinstance(image, torch.Tensor):
            tensor = image.detach().cpu()
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)
            if tensor.dim() != 3:
                raise ValueError(f"Unsupported tensor shape for stats: {tuple(tensor.shape)}")
            if tensor.max().item() > 1.0:
                tensor = tensor / 255.0
        else:
            raise TypeError(f"Unsupported image type for stats: {type(image)}")

        tensor = tensor.to(dtype=torch.float64)
        channel_sum += tensor.sum(dim=(1, 2))
        channel_sq_sum += (tensor ** 2).sum(dim=(1, 2))
        pixel_count += tensor.shape[1] * tensor.shape[2]

    mean = channel_sum / pixel_count
    variance = (channel_sq_sum / pixel_count) - mean ** 2
    variance = torch.clamp(variance, min=0.0)
    std = torch.sqrt(variance)
    return mean.tolist(), std.tolist()


# ============================================================================
# TRANSFORM PIPELINE BUILDERS
# ============================================================================

class TransformPipeline:
    """
    Factory for creating image transformation pipelines.
    
    Provides separate pipelines for:
    - Training: includes augmentation
    - Validation: no augmentation
    - Testing: no augmentation (strict)
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        normalization_mean: List[float] = IMAGENET_MEAN,
        normalization_std: List[float] = IMAGENET_STD,
        augmentation_config: Optional[AugmentationConfig] = None,
    ):
        """
        Initialize transform pipeline builder.
        
        Args:
            image_size: Target image size (height, width)
            normalization_mean: Normalization mean per channel
            normalization_std: Normalization std per channel
            augmentation_config: Augmentation configuration (if None, uses defaults)
        """
        self.image_size = image_size
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        self.aug_config = augmentation_config or AugmentationConfig()
        
        logger.info(f"Transform pipeline initialized: "
                   f"size={image_size}, "
                   f"mean={normalization_mean}, "
                   f"std={normalization_std}")
    
    def get_train_transforms(self) -> transforms.Compose:
        """
        Get training transforms with data augmentation.
        
        Augmentation strategy:
        - Geometric: random flip, rotation, affine transform
        - Color: brightness, contrast, saturation, hue jitter
        - Final: resize, normalization
        
        Returns:
            Composition of training transforms
        """
        augmentation_list = [
            # Geometric augmentations
            transforms.RandomHorizontalFlip(p=self.aug_config.random_horizontal_flip_prob),
            transforms.RandomVerticalFlip(p=self.aug_config.random_vertical_flip_prob),
            transforms.RandomRotation(
                degrees=self.aug_config.random_rotation_degrees,
                fill=0  # Fill with black for medical images
            ),
            
            # Affine transform (translation + shear)
            transforms.RandomAffine(
                degrees=0,  # Already handled by RandomRotation
                translate=self.aug_config.affine_translate_percent,
                shear=self.aug_config.shear_degrees,
                fill=0
            ),
            
            # Colour augmentation remains conservative to preserve diagnostic cues.
            transforms.ColorJitter(
                brightness=self.aug_config.brightness_factor,
                contrast=self.aug_config.contrast_factor,
                saturation=self.aug_config.saturation_factor,
                hue=self.aug_config.hue_factor,
            ),
        ]
        
        # Mild geometric perturbation approximates acquisition-angle changes.
        # Extreme zoom (> 1.3x) is intentionally excluded because it can remove
        # lesion context and distort apparent lesion scale.
        augmentation_list.append(
            transforms.RandomPerspective(
                distortion_scale=self.aug_config.perspective_distortion_scale,
                p=0.3,
                fill=0
            )
        )

        # Random erasing / cutout is intentionally excluded because it can cover
        # the lesion itself or diagnostically important surrounding tissue.
        
        # Standard preprocessing
        augmentation_list.extend([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.normalization_mean,
                std=self.normalization_std
            ),
        ])
        
        logger.info("Training transforms: augmentation + normalization")
        return transforms.Compose(augmentation_list)
    
    def get_val_transforms(self) -> transforms.Compose:
        """
        Get validation transforms without augmentation.
        
        Returns:
            Composition of validation transforms
        """
        val_transforms = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.normalization_mean,
                std=self.normalization_std
            ),
        ])
        
        logger.info("Validation transforms: resize + normalize (no augmentation)")
        return val_transforms
    
    def get_test_transforms(self) -> transforms.Compose:
        """
        Get test transforms without augmentation.
        
        Same as validation transforms - strict preprocessing only.
        
        Returns:
            Composition of test transforms
        """
        test_transforms = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.normalization_mean,
                std=self.normalization_std
            ),
        ])
        
        logger.info("Test transforms: resize + normalize (no augmentation)")
        return test_transforms
    
    def get_transforms_dict(self) -> Dict[str, transforms.Compose]:
        """
        Get all transforms as a dictionary.
        
        Returns:
            Dictionary with keys 'train', 'val', 'test'
        """
        return {
            'train': self.get_train_transforms(),
            'val': self.get_val_transforms(),
            'test': self.get_test_transforms(),
        }


# ============================================================================
# WRAPPED DATASET CLASSES WITH TRANSFORMS
# ============================================================================

class TransformedDataset(torch.utils.data.Dataset):
    """
    Wrapper around BaseDataset that applies transforms.
    
    This allows different transforms for train/val/test while using
    the same underlying dataset.
    """
    
    def __init__(
        self,
        base_dataset,
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Initialize transformed dataset wrapper.
        
        Args:
            base_dataset: Dataset from data_loader module
            transform: Transform to apply to images
        """
        self.base_dataset = base_dataset
        self.transform = transform
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict]:
        """
        Get item with transform applied.

        Args:
            idx: Sample index

        Returns:
            Tuple of (transformed_image, label, metadata)
        """
        image, label, metadata = self.base_dataset[idx]

        # FIX (double-transform bug): BaseDataset now returns a raw PIL Image.
        # We apply the transform directly — no lossy denormalize round-trip needed.
        if self.transform is not None:
            image = self.transform(image)

        return image, label, metadata


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_train_val_transforms(
    image_size: Tuple[int, int] = (224, 224),
    augmentation_config: Optional[AugmentationConfig] = None,
) -> Dict[str, transforms.Compose]:
    """
    Create training and validation transforms.
    
    Args:
        image_size: Target image size
        augmentation_config: Augmentation configuration
        
    Returns:
        Dictionary with 'train' and 'val' transforms
    """
    pipeline = TransformPipeline(
        image_size=image_size,
        augmentation_config=augmentation_config
    )
    
    return {
        'train': pipeline.get_train_transforms(),
        'val': pipeline.get_val_transforms(),
    }


def create_all_transforms(
    image_size: Tuple[int, int] = (224, 224),
    augmentation_config: Optional[AugmentationConfig] = None,
) -> Dict[str, transforms.Compose]:
    """
    Create all transforms (train, val, test).
    
    Args:
        image_size: Target image size
        augmentation_config: Augmentation configuration
        
    Returns:
        Dictionary with 'train', 'val', and 'test' transforms
    """
    pipeline = TransformPipeline(
        image_size=image_size,
        augmentation_config=augmentation_config
    )
    
    return pipeline.get_transforms_dict()


def get_normalization_stats(normalization_type: str = 'imagenet') -> Tuple[List, List]:
    """
    Get normalization statistics.
    
    Args:
        normalization_type: 'imagenet' or 'medical'
        
    Returns:
        Tuple of (mean, std)
    """
    if normalization_type.lower() == 'imagenet':
        return IMAGENET_MEAN, IMAGENET_STD
    elif normalization_type.lower() == 'medical':
        return MEDICAL_MEAN, MEDICAL_STD
    else:
        raise ValueError(f"Unknown normalization type: {normalization_type}")


def compute_dataset_normalization_stats(
    dataloader,
    num_batches: int = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute normalization statistics (mean and std) from dataset.
    
    Use this to compute dataset-specific normalization if working with
    a highly specialized medical imaging domain.
    
    Args:
        dataloader: PyTorch DataLoader with unnormalized images
        num_batches: Number of batches to use (None = all)
        
    Returns:
        Tuple of (mean, std) tensors per channel
    """
    mean = torch.zeros(3)
    std = torch.zeros(3)
    batch_count = 0
    
    for batch_idx, (images, labels, metadata) in enumerate(dataloader):
        if num_batches is not None and batch_idx >= num_batches:
            break
        
        # images shape: (batch_size, 3, H, W)
        for i in range(3):  # RGB channels
            mean[i] += images[:, i, :, :].mean()
            std[i] += images[:, i, :, :].std()
        
        batch_count += 1
    
    mean /= batch_count
    std /= batch_count
    
    logger.info(f"Computed dataset normalization: mean={mean}, std={std}")
    return mean, std


# ============================================================================
# AUGMENTATION VISUALIZATION HELPERS
# ============================================================================

def visualize_augmentations(
    image: torch.Tensor,
    augmentation_config: AugmentationConfig,
    num_samples: int = 9,
) -> List[torch.Tensor]:
    """
    Create multiple augmented versions of the same image for visualization.
    
    Useful for verifying augmentation strategy is clinically appropriate.
    
    Args:
        image: Input image tensor (C, H, W)
        augmentation_config: Augmentation configuration
        num_samples: Number of augmented versions to create
        
    Returns:
        List of augmented image tensors
    """
    pipeline = TransformPipeline(augmentation_config=augmentation_config)
    train_transform = pipeline.get_train_transforms()
    
    # Convert tensor back to PIL
    image_np = image.numpy().transpose(1, 2, 0)
    image_np = image_np * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    image_np = (image_np * 255).astype(np.uint8)
    
    from PIL import Image
    image_pil = Image.fromarray(image_np, mode='RGB')
    
    # Generate augmented versions
    augmented_images = []
    for _ in range(num_samples):
        augmented = train_transform(image_pil)
        augmented_images.append(augmented)
    
    return augmented_images


if __name__ == "__main__":
    """Demonstrate preprocessing pipeline."""
    
    # Example augmentation config
    aug_config = AugmentationConfig(
        random_horizontal_flip_prob=0.5,
        random_vertical_flip_prob=0.3,
        random_rotation_degrees=15,
        brightness_factor=0.1,
        contrast_factor=0.1,
    )
    
    # Create transforms
    transforms_dict = create_all_transforms(
        image_size=(224, 224),
        augmentation_config=aug_config
    )
    
    print("Transform pipelines created:")
    for split, transform in transforms_dict.items():
        print(f"\n{split.upper()}:")
        print(transform)
    
    # Print normalization stats
    mean, std = get_normalization_stats('imagenet')
    print(f"\nImageNet Normalization:")
    print(f"  Mean: {mean}")
    print(f"  Std: {std}")
