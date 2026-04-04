"""
Multi-Dataset Clinical AI Data Loading Module

This module provides a robust, reusable data loading interface for clinical AI projects
that work with multiple datasets (public and clinical). It handles:
- Image loading and preprocessing
- Label standardization (binary classification)
- Missing data handling
- Consistent metadata management
- Prevention of data leakage between datasets

Author: Clinical AI Development
Date: 2026
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass
import logging

import pandas as pd
import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES AND CONFIGURATION
# ============================================================================

@dataclass
class ImageMetadata:
    """
    Container for image metadata including diagnosis information and lesion details.
    
    Attributes:
        image_id: Unique identifier for the image
        image_name: Filename of the image
        label: Binary label (0 or 1) - gold standard diagnosis
        clinician_diagnosis: Optional clinician's initial diagnosis (may differ from label)
        lesion_type: Optional classification of lesion type
        location: Optional anatomical location
        dataset_source: Source dataset identifier (for tracking data origin)
        patient_id: Optional patient identifier — required for patient-level splitting to
            prevent leakage when a patient has multiple images. Populate from the CSV
            column 'patient_id' (or equivalent) when available.
    """
    image_id: str
    image_name: str
    label: int
    clinician_diagnosis: Optional[int] = None
    lesion_type: Optional[str] = None
    location: Optional[str] = None
    dataset_source: Optional[str] = None
    # FIX (patient leakage): track patient identity so split_clinical_dataset can
    # group all images from the same patient into the same split.
    patient_id: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert metadata to dictionary for serialization."""
        return {
            'image_id': self.image_id,
            'image_name': self.image_name,
            'label': self.label,
            'clinician_diagnosis': self.clinician_diagnosis,
            'lesion_type': self.lesion_type,
            'location': self.location,
            'dataset_source': self.dataset_source,
            'patient_id': self.patient_id,
        }


# ============================================================================
# BASE DATASET CLASS
# ============================================================================

class BaseDataset(Dataset):
    """
    Abstract base class for clinical imaging datasets.
    
    Provides common functionality for loading, validating, and preprocessing
    images from clinical datasets.
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        images_dir: str = 'images',
        image_size: Tuple[int, int] = (224, 224),
        image_extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg'),
        normalize: bool = True,
    ):
        """
        Initialize the base dataset.
        
        Args:
            root_dir: Root directory of the dataset
            images_dir: Subdirectory containing images (default: 'images')
            image_size: Target size for images (default: 224x224)
            image_extensions: Allowed image file extensions
            normalize: Whether to normalize images to [0, 1]
        """
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / images_dir
        self.image_size = image_size
        self.image_extensions = image_extensions
        self.normalize = normalize
        
        # FIX (double-transform bug): BaseDataset no longer applies Normalize here.
        # _load_image returns a raw PIL Image so that TransformedDataset can apply
        # the correct train/val/test transform once — avoiding the lossy
        # float-tensor → uint8-PIL denormalize round-trip that was here before.
        
        # Container for samples
        self.samples: List[Tuple[str, ImageMetadata]] = []
        
        # Validate directories
        self._validate_directories()
    
    def _validate_directories(self) -> None:
        """Validate that required directories exist."""
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")
        
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        logger.info(f"Dataset initialized at: {self.root_dir}")
    
    def _load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """
        Load an image as a PIL Image (no normalization applied here).

        All resizing and normalization is handled downstream by TransformedDataset
        so that train/val/test pipelines each receive the correct transforms exactly
        once — without any lossy intermediate denormalization step.

        Args:
            image_path: Path to the image file

        Returns:
            PIL.Image.Image: Raw RGB image

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be loaded
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            return Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to load image {image_path}: {str(e)}")
    
    def _validate_label(self, label: Union[int, float, str]) -> int:
        """
        Validate and standardize label to binary format.
        
        Args:
            label: Label value in any format
            
        Returns:
            int: Binary label (0 or 1)
            
        Raises:
            ValueError: If label cannot be converted to binary
        """
        try:
            # Convert to float first to handle string inputs
            label_float = float(label)
            
            # Handle continuous values (e.g., probabilities or confidence scores)
            # Threshold at 0.5
            if label_float < 0 or label_float > 1:
                raise ValueError(f"Label value out of range [0, 1]: {label_float}")
            
            # Convert to binary
            binary_label = int(round(label_float))
            
            return binary_label
        
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert label '{label}' to binary format: {str(e)}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[Image.Image, int, Dict]:
        """
        Get a sample from the dataset.

        Returns a raw PIL Image so that the correct transform (train/val/test)
        can be applied exactly once by TransformedDataset. Wrap this dataset
        with TransformedDataset before creating a DataLoader.

        Args:
            idx: Index of the sample

        Returns:
            Tuple containing:
                - image: Raw PIL Image (RGB)
                - label: Binary classification label (0 or 1)
                - metadata: Dictionary with metadata information
        """
        image_path, metadata = self.samples[idx]

        try:
            image = self._load_image(image_path)
            return image, metadata.label, metadata.to_dict()

        except Exception as e:
            logger.error(f"Error loading sample {idx}: {str(e)}")
            raise


# ============================================================================
# PUBLIC DATASET CLASS
# ============================================================================

class PublicDataset(BaseDataset):
    """
    Dataset class for public datasets with potentially noisy labels.
    
    Handles datasets from public sources (e.g., oral lesion databases) that may
    have inconsistent annotation quality or missing information.
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        labels_file: str = 'labels.csv',
        image_dir_name: str = 'images',
        label_column: str = 'label',
        image_name_column: str = 'image_name',
        **kwargs
    ):
        """
        Initialize a public dataset.
        
        Args:
            root_dir: Root directory of the dataset
            labels_file: Name of the CSV file containing labels
            image_dir_name: Name of the subdirectory containing images
            label_column: Column name for labels in CSV
            image_name_column: Column name for image filenames in CSV
            **kwargs: Additional arguments for BaseDataset
        """
        super().__init__(root_dir=root_dir, images_dir=image_dir_name, **kwargs)
        
        self.labels_file = self.root_dir / labels_file
        self.label_column = label_column
        self.image_name_column = image_name_column
        self.dataset_source = "PUBLIC"
        
        # Load and validate data
        self._load_public_dataset()
    
    def _load_public_dataset(self) -> None:
        """
        Load public dataset from CSV file.
        
        Reads labels from CSV, validates data, and constructs sample list.
        Handles missing values gracefully.
        """
        if not self.labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_file}")
        
        try:
            df = pd.read_csv(self.labels_file)
        except Exception as e:
            raise ValueError(f"Failed to read labels file: {str(e)}")
        
        # Validate required columns
        if self.label_column not in df.columns:
            raise ValueError(
                f"Column '{self.label_column}' not found in labels file. "
                f"Available columns: {df.columns.tolist()}"
            )
        
        if self.image_name_column not in df.columns:
            raise ValueError(
                f"Column '{self.image_name_column}' not found in labels file. "
                f"Available columns: {df.columns.tolist()}"
            )
        
        # Load samples
        loaded_count = 0
        skipped_count = 0
        
        for idx, row in df.iterrows():
            try:
                image_name = str(row[self.image_name_column])
                image_path = self.images_dir / image_name
                
                # Skip missing images
                if not image_path.exists():
                    logger.warning(f"Image not found: {image_path}")
                    skipped_count += 1
                    continue
                
                # Validate and standardize label
                label = self._validate_label(row[self.label_column])
                
                # Create metadata
                metadata = ImageMetadata(
                    image_id=str(idx),
                    image_name=image_name,
                    label=label,
                    dataset_source=self.dataset_source,
                )
                
                self.samples.append((str(image_path), metadata))
                loaded_count += 1
            
            except (KeyError, ValueError) as e:
                logger.warning(f"Skipped row {idx}: {str(e)}")
                skipped_count += 1
                continue
        
        logger.info(
            f"Loaded {loaded_count} samples from public dataset at {self.root_dir}. "
            f"Skipped: {skipped_count}"
        )
        
        if loaded_count == 0:
            raise ValueError(f"No valid samples found in {self.root_dir}")


# ============================================================================
# CLINICAL DATASET CLASS
# ============================================================================

class ClinicalDataset(BaseDataset):
    """
    Dataset class for clinical datasets with biopsy-confirmed diagnoses.
    
    Handles clinical datasets with gold-standard labels (biopsy-confirmed diagnosis)
    and optional clinician annotations. Prevents data leakage by tracking data source.
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        metadata_file: str = 'metadata.csv',
        image_dir_name: str = 'images',
        id_column: str = 'id',
        image_name_column: str = 'image_name',
        label_column: str = 'biopsy_diagnosis',
        clinician_diagnosis_column: Optional[str] = 'clinician_diagnosis',
        lesion_type_column: Optional[str] = 'lesion_type',
        location_column: Optional[str] = 'location',
        **kwargs
    ):
        """
        Initialize a clinical dataset.
        
        Args:
            root_dir: Root directory of the dataset
            metadata_file: Name of the CSV file containing metadata
            image_dir_name: Name of the subdirectory containing images
            id_column: Column name for image ID
            image_name_column: Column name for image filenames
            label_column: Column name for biopsy diagnosis (ground truth)
            clinician_diagnosis_column: Column name for clinician diagnosis (optional)
            lesion_type_column: Column name for lesion type (optional)
            location_column: Column name for lesion location (optional)
            **kwargs: Additional arguments for BaseDataset
        """
        super().__init__(root_dir=root_dir, images_dir=image_dir_name, **kwargs)
        
        self.metadata_file = self.root_dir / metadata_file
        self.id_column = id_column
        self.image_name_column = image_name_column
        self.label_column = label_column
        self.clinician_diagnosis_column = clinician_diagnosis_column
        self.lesion_type_column = lesion_type_column
        self.location_column = location_column
        self.dataset_source = "CLINICAL"
        
        # Load and validate data
        self._load_clinical_dataset()
    
    def _load_clinical_dataset(self) -> None:
        """
        Load clinical dataset from metadata file.
        
        Reads metadata from CSV, validates biopsy diagnosis (gold standard),
        and handles optional fields gracefully.
        """
        if not self.metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_file}")
        
        try:
            df = pd.read_csv(self.metadata_file)
        except Exception as e:
            raise ValueError(f"Failed to read metadata file: {str(e)}")
        
        # Validate required columns
        required_columns = [self.label_column, self.image_name_column]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(
                    f"Required column '{col}' not found in metadata. "
                    f"Available columns: {df.columns.tolist()}"
                )
        
        # Load samples
        loaded_count = 0
        skipped_count = 0
        
        for idx, row in df.iterrows():
            try:
                image_name = str(row[self.image_name_column])
                image_path = self.images_dir / image_name
                
                # Skip missing images
                if not image_path.exists():
                    logger.warning(f"Image not found: {image_path}")
                    skipped_count += 1
                    continue
                
                # Validate and standardize biopsy diagnosis (gold standard)
                label = self._validate_label(row[self.label_column])
                
                # Extract optional clinician diagnosis
                clinician_diagnosis = None
                if self.clinician_diagnosis_column and \
                   self.clinician_diagnosis_column in df.columns:
                    try:
                        if pd.notna(row[self.clinician_diagnosis_column]):
                            clinician_diagnosis = self._validate_label(
                                row[self.clinician_diagnosis_column]
                            )
                    except ValueError:
                        logger.debug(
                            f"Could not parse clinician diagnosis for {image_name}"
                        )
                
                # Extract optional metadata fields
                lesion_type = None
                if self.lesion_type_column and self.lesion_type_column in df.columns:
                    lesion_type = str(row[self.lesion_type_column]) \
                        if pd.notna(row[self.lesion_type_column]) else None
                
                location = None
                if self.location_column and self.location_column in df.columns:
                    location = str(row[self.location_column]) \
                        if pd.notna(row[self.location_column]) else None
                
                # Extract patient_id (required for patient-level splitting)
                patient_id = None
                if 'patient_id' in df.columns and pd.notna(row.get('patient_id')):
                    patient_id = str(row['patient_id'])
                
                # Create metadata
                image_id = str(row[self.id_column]) if self.id_column in df.columns \
                    else str(idx)
                
                metadata = ImageMetadata(
                    image_id=image_id,
                    image_name=image_name,
                    label=label,
                    clinician_diagnosis=clinician_diagnosis,
                    lesion_type=lesion_type,
                    location=location,
                    dataset_source=self.dataset_source,
                    patient_id=patient_id,
                )
                
                self.samples.append((str(image_path), metadata))
                loaded_count += 1
            
            except (KeyError, ValueError) as e:
                logger.warning(f"Skipped row {idx}: {str(e)}")
                skipped_count += 1
                continue
        
        logger.info(
            f"Loaded {loaded_count} samples from clinical dataset at {self.root_dir}. "
            f"Skipped: {skipped_count}"
        )
        
        if loaded_count == 0:
            raise ValueError(f"No valid samples found in {self.root_dir}")
    
    def diagnostic_agreement(self) -> float:
        """
        Calculate agreement between clinician and biopsy diagnoses.
        
        Returns:
            float: Proportion of cases where clinician diagnosis matches biopsy.
        """
        if not any(m.clinician_diagnosis is not None for _, m in self.samples):
            logger.warning("No clinician diagnoses available")
            return np.nan
        
        agreements = 0
        total_with_clinician = 0
        
        for _, metadata in self.samples:
            if metadata.clinician_diagnosis is not None:
                total_with_clinician += 1
                if metadata.clinician_diagnosis == metadata.label:
                    agreements += 1
        
        if total_with_clinician == 0:
            return np.nan
        
        return agreements / total_with_clinician


# ============================================================================
# DATASET FACTORY FUNCTIONS
# ============================================================================

def get_public_dataset(
    dataset_path: Union[str, Path],
    labels_file: str = 'labels.csv',
    image_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
) -> PublicDataset:
    """
    Create and return a public dataset instance.
    
    These datasets may contain noisy labels and are typically used for
    pretraining or supplementary training.
    
    Args:
        dataset_path: Path to the public dataset directory
        labels_file: Name of the labels CSV file
        image_size: Target image size
        normalize: Whether to normalize images
        
    Returns:
        PublicDataset: Initialized public dataset instance
    """
    return PublicDataset(
        root_dir=dataset_path,
        labels_file=labels_file,
        image_size=image_size,
        normalize=normalize,
    )


def get_clinical_dataset(
    dataset_path: Union[str, Path],
    metadata_file: str = 'metadata.csv',
    image_size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
) -> ClinicalDataset:
    """
    Create and return a clinical dataset instance.
    
    These datasets contain biopsy-confirmed diagnoses (gold standard) and should be
    protected to prevent data leakage.
    
    Args:
        dataset_path: Path to the clinical dataset directory
        metadata_file: Name of the metadata CSV file
        image_size: Target image size
        normalize: Whether to normalize images
        
    Returns:
        ClinicalDataset: Initialized clinical dataset instance
    """
    return ClinicalDataset(
        root_dir=dataset_path,
        metadata_file=metadata_file,
        image_size=image_size,
        normalize=normalize,
    )



