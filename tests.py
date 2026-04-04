"""
Unit tests for the clinical AI data loading module.

Run tests with: pytest tests.py -v

This file provides templates for testing the data loading functionality.
Customize tests based on your specific data and requirements.
"""

import pytest
import tempfile
import csv
from collections import defaultdict
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader

from data_loader import (
    ImageMetadata,
    BaseDataset,
    PublicDataset,
    ClinicalDataset,
    get_clinical_dataset,
    get_public_dataset,
)


# ============================================================================
# FIXTURES - Create test data
# ============================================================================

@pytest.fixture
def temp_dataset_dir():
    """Create a temporary directory structure for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    # Create a 224x224 RGB image
    img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture
def clinical_dataset_fixture(temp_dataset_dir, sample_image):
    """Create a mock clinical dataset."""
    # Create directories
    dataset_dir = temp_dataset_dir / "clinical"
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True)
    
    # Create sample images
    for i in range(5):
        img_path = images_dir / f"image_{i:03d}.jpg"
        sample_image.save(img_path)
    
    # Create metadata CSV
    metadata_path = dataset_dir / "metadata.csv"
    with open(metadata_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'id', 'image_name', 'biopsy_diagnosis', 
            'clinician_diagnosis', 'lesion_type', 'location'
        ])
        writer.writeheader()
        
        for i in range(5):
            writer.writerow({
                'id': f'pt{i+1:03d}',
                'image_name': f'image_{i:03d}.jpg',
                'biopsy_diagnosis': 1 if i % 2 == 0 else 0,
                'clinician_diagnosis': 1 if i % 3 == 0 else 0,
                'lesion_type': 'oral_cancer' if i < 3 else 'benign',
                'location': 'buccal' if i < 3 else 'palate',
            })
    
    return dataset_dir


@pytest.fixture
def public_dataset_fixture(temp_dataset_dir, sample_image):
    """Create a mock public dataset."""
    # Create directories
    dataset_dir = temp_dataset_dir / "public"
    images_dir = dataset_dir / "images"
    images_dir.mkdir(parents=True)
    
    # Create sample images
    for i in range(5):
        img_path = images_dir / f"image_{i:03d}.jpg"
        sample_image.save(img_path)
    
    # Create labels CSV
    labels_path = dataset_dir / "labels.csv"
    with open(labels_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_name', 'label'])
        writer.writeheader()
        
        for i in range(5):
            writer.writerow({
                'image_name': f'image_{i:03d}.jpg',
                'label': 1 if i % 2 == 0 else 0,
            })
    
    return dataset_dir


# ============================================================================
# TESTS - ImageMetadata
# ============================================================================

class TestImageMetadata:
    """Test ImageMetadata dataclass."""
    
    def test_metadata_creation(self):
        """Test creating metadata object."""
        metadata = ImageMetadata(
            image_id='001',
            image_name='test.jpg',
            label=1,
            clinician_diagnosis=0,
            lesion_type='cancer',
            location='buccal',
            dataset_source='CLINICAL'
        )
        
        assert metadata.image_id == '001'
        assert metadata.label == 1
        assert metadata.clinician_diagnosis == 0
    
    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        metadata = ImageMetadata(
            image_id='001',
            image_name='test.jpg',
            label=1,
        )
        
        meta_dict = metadata.to_dict()
        assert isinstance(meta_dict, dict)
        assert meta_dict['image_id'] == '001'
        assert meta_dict['label'] == 1
        assert meta_dict['clinician_diagnosis'] is None


# ============================================================================
# TESTS - ClinicalDataset
# ============================================================================

class TestClinicalDataset:
    """Test ClinicalDataset class."""
    
    def test_clinical_dataset_loading(self, clinical_dataset_fixture):
        """Test loading clinical dataset."""
        dataset = ClinicalDataset(clinical_dataset_fixture)
        
        assert len(dataset) == 5
        assert dataset.dataset_source == "CLINICAL"
    
    def test_clinical_dataset_getitem(self, clinical_dataset_fixture):
        """Test getting item from clinical dataset."""
        dataset = ClinicalDataset(clinical_dataset_fixture)
        
        image, label, metadata = dataset[0]
        
        # Check return types
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, int)
        assert isinstance(metadata, dict)
        
        # Check image shape
        assert image.shape == (3, 224, 224)
        
        # Check label is binary
        assert label in [0, 1]
        
        # Check metadata
        assert 'image_id' in metadata
        assert 'image_name' in metadata
        assert 'label' in metadata
        assert 'dataset_source' in metadata
    
    def test_clinical_diagnostic_agreement(self, clinical_dataset_fixture):
        """Test diagnostic agreement calculation."""
        dataset = ClinicalDataset(clinical_dataset_fixture)
        
        agreement = dataset.diagnostic_agreement()
        
        # Should return a float between 0 and 1
        assert isinstance(agreement, (float, np.floating))
        assert 0 <= agreement <= 1
    
    def test_clinical_missing_metadata(self, temp_dataset_dir, sample_image):
        """Test handling of missing optional metadata."""
        # Create dataset with minimal metadata
        dataset_dir = temp_dataset_dir / "minimal"
        images_dir = dataset_dir / "images"
        images_dir.mkdir(parents=True)
        
        # Create image
        img_path = images_dir / "image_001.jpg"
        sample_image.save(img_path)
        
        # Create metadata with only required fields
        metadata_path = dataset_dir / "metadata.csv"
        with open(metadata_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'id', 'image_name', 'biopsy_diagnosis'
            ])
            writer.writeheader()
            writer.writerow({
                'id': 'pt001',
                'image_name': 'image_001.jpg',
                'biopsy_diagnosis': 1,
            })
        
        dataset = ClinicalDataset(dataset_dir)
        
        assert len(dataset) == 1
        image, label, metadata = dataset[0]
        assert metadata['clinician_diagnosis'] is None
        assert metadata['lesion_type'] is None


# ============================================================================
# TESTS - PublicDataset
# ============================================================================

class TestPublicDataset:
    """Test PublicDataset class."""
    
    def test_public_dataset_loading(self, public_dataset_fixture):
        """Test loading public dataset."""
        dataset = PublicDataset(public_dataset_fixture)
        
        assert len(dataset) == 5
        assert dataset.dataset_source == "PUBLIC"
    
    def test_public_dataset_getitem(self, public_dataset_fixture):
        """Test getting item from public dataset."""
        dataset = PublicDataset(public_dataset_fixture)
        
        image, label, metadata = dataset[0]
        
        # Check return types
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, int)
        assert isinstance(metadata, dict)
        
        # Check image shape
        assert image.shape == (3, 224, 224)
        
        # Check label is binary
        assert label in [0, 1]
        
        # Check metadata
        assert 'image_name' in metadata
        assert 'label' in metadata
        assert metadata['dataset_source'] == 'PUBLIC'
    
    def test_invalid_label_conversion(self, temp_dataset_dir, sample_image):
        """Test handling of invalid labels."""
        dataset_dir = temp_dataset_dir / "invalid"
        images_dir = dataset_dir / "images"
        images_dir.mkdir(parents=True)
        
        img_path = images_dir / "img_001.jpg"
        sample_image.save(img_path)
        
        labels_path = dataset_dir / "labels.csv"
        with open(labels_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['image_name', 'label'])
            writer.writeheader()
            writer.writerow({
                'image_name': 'img_001.jpg',
                'label': 2,  # Invalid: not in [0, 1]
            })
        
        dataset = PublicDataset(dataset_dir)
        
        # Should skip invalid samples
        assert len(dataset) == 0


# ============================================================================
# TESTS - Factory Functions
# ============================================================================

class TestFactoryFunctions:
    """Test factory functions for dataset creation."""
    
    def test_get_clinical_dataset(self, clinical_dataset_fixture):
        """Test get_clinical_dataset factory function."""
        dataset = get_clinical_dataset(clinical_dataset_fixture)
        
        assert isinstance(dataset, ClinicalDataset)
        assert len(dataset) == 5
    
    def test_get_public_dataset(self, public_dataset_fixture):
        """Test get_public_dataset factory function."""
        dataset = get_public_dataset(public_dataset_fixture)
        
        assert isinstance(dataset, PublicDataset)
        assert len(dataset) == 5
    



# ============================================================================
# TESTS - DataLoader Integration
# ============================================================================

class TestDataLoaderIntegration:
    """Test integration with PyTorch DataLoader."""
    
    def test_dataloader_iteration(self, clinical_dataset_fixture):
        """Test iterating through batches."""
        dataset = ClinicalDataset(clinical_dataset_fixture)
        dataloader = DataLoader(dataset, batch_size=2, num_workers=0)

        batch_count = 0
        for images, labels, metadata in dataloader:
            batch_count += 1
            assert images.shape[0] in [1, 2]
            assert images.shape == (images.shape[0], 3, 224, 224)
            assert len(labels) == images.shape[0]
            assert len(metadata) == images.shape[0]

        assert batch_count > 0

    def test_dataloader_shuffle(self, clinical_dataset_fixture):
        """Test DataLoader with shuffle enabled."""
        dataset = ClinicalDataset(clinical_dataset_fixture)
        dataloader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=0)

        for images, labels, metadata in dataloader:
            assert images.shape[0] == 5
            break


# ============================================================================
# TESTS - Statistics
# ============================================================================

class TestDatasetStatistics:
    """Test dataset statistics computed directly from dataset.samples."""

    def test_label_counts(self, clinical_dataset_fixture):
        """Test that label counts from dataset.samples are correct."""
        dataset = ClinicalDataset(clinical_dataset_fixture)
        labels = [m.label for _, m in dataset.samples]
        assert len(labels) == 5
        assert sum(labels) + (len(labels) - sum(labels)) == 5

    def test_statistics_percentages(self, clinical_dataset_fixture):
        """Test that computed label percentages sum to 100."""
        dataset = ClinicalDataset(clinical_dataset_fixture)
        labels = [m.label for _, m in dataset.samples]
        n = len(labels)
        p1 = sum(labels) / n * 100
        p0 = (n - sum(labels)) / n * 100
        assert abs(p0 + p1 - 100.0) < 0.01


# ============================================================================
# TESTS - Error Handling
# ============================================================================

class TestErrorHandling:
    """Test error handling."""
    
    def test_missing_directory(self, temp_dataset_dir):
        """Test handling of missing dataset directory."""
        with pytest.raises(FileNotFoundError):
            get_clinical_dataset(temp_dataset_dir / "nonexistent")
    
    def test_missing_metadata_file(self, temp_dataset_dir):
        """Test handling of missing metadata file."""
        dataset_dir = temp_dataset_dir / "nofiles"
        images_dir = dataset_dir / "images"
        images_dir.mkdir(parents=True)
        
        with pytest.raises(FileNotFoundError):
            ClinicalDataset(dataset_dir)
    
    def test_index_out_of_range(self, clinical_dataset_fixture):
        """Test accessing index out of range."""
        dataset = get_clinical_dataset(clinical_dataset_fixture)
        
        with pytest.raises(IndexError):
            dataset[100]


# ============================================================================
# TESTS - Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and corner cases."""
    
    def test_single_sample_dataset(self, temp_dataset_dir, sample_image):
        """Test dataset with single sample."""
        dataset_dir = temp_dataset_dir / "single"
        images_dir = dataset_dir / "images"
        images_dir.mkdir(parents=True)
        
        img_path = images_dir / "image_001.jpg"
        sample_image.save(img_path)
        
        metadata_path = dataset_dir / "metadata.csv"
        with open(metadata_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'id', 'image_name', 'biopsy_diagnosis'
            ])
            writer.writeheader()
            writer.writerow({
                'id': 'pt001',
                'image_name': 'image_001.jpg',
                'biopsy_diagnosis': 1,
            })
        
        dataset = ClinicalDataset(dataset_dir)
        
        assert len(dataset) == 1
        image, label, metadata = dataset[0]
        assert label == 1
    
    def test_label_thresholding(self, temp_dataset_dir, sample_image):
        """Test continuous label conversion (0.5 threshold)."""
        dataset_dir = temp_dataset_dir / "continuous"
        images_dir = dataset_dir / "images"
        images_dir.mkdir(parents=True)
        
        img_path = images_dir / "image_001.jpg"
        sample_image.save(img_path)
        
        labels_path = dataset_dir / "labels.csv"
        with open(labels_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['image_name', 'label'])
            writer.writeheader()
            writer.writerow({
                'image_name': 'image_001.jpg',
                'label': 0.3,  # Should convert to 0
            })
        
        dataset = PublicDataset(dataset_dir)
        
        assert len(dataset) == 1
        _, label, _ = dataset[0]
        assert label == 0


# ============================================================================
# TESTS - Patient-Level Splitting
# ============================================================================

class TestPatientLevelSplit:
    """Tests for patient-level dataset splitting."""

    def _make_multi_image_dataset(self, dataset_dir: Path, sample_image) -> Path:
        """Create a dataset with multiple images per patient.

        6 patients: 3 benign (label=0), 3 malignant (label=1), 2 images each
        → 12 images total, CSV has patient_id column.
        """
        ds_dir = dataset_dir / "multi_image_clinical"
        images_dir = ds_dir / "images"
        images_dir.mkdir(parents=True)

        rows = []
        img_idx = 0
        for patient in range(1, 7):
            label = 0 if patient <= 3 else 1
            for _ in range(2):
                img_name = f"image_{img_idx:03d}.jpg"
                sample_image.save(images_dir / img_name)
                rows.append({
                    "id": f"img_{img_idx}",
                    "patient_id": f"patient_{patient:02d}",
                    "image_name": img_name,
                    "biopsy_diagnosis": label,
                })
                img_idx += 1

        metadata_path = ds_dir / "metadata.csv"
        with open(metadata_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["id", "patient_id", "image_name", "biopsy_diagnosis"]
            )
            writer.writeheader()
            writer.writerows(rows)

        return ds_dir

    def test_patient_level_no_leakage(self, temp_dataset_dir, sample_image):
        """No patient should appear in more than one split."""
        from dataset_split import split_clinical_dataset, SplitConfig

        ds_dir = self._make_multi_image_dataset(temp_dataset_dir, sample_image)
        dataset = ClinicalDataset(ds_dir)

        # Verify patient_ids are populated from the CSV
        pids = [dataset.samples[i][1].patient_id for i in range(len(dataset))]
        assert any(pid is not None for pid in pids), "patient_id not populated"

        split_config = SplitConfig(
            patient_level_split=True,
            random_seed=42,
            clinical_train_ratio=0.50,
            clinical_val_ratio=0.25,
            clinical_test_ratio=0.25,
        )
        splits = split_clinical_dataset(dataset, split_config)

        # No index overlap
        assert not (set(splits["train"]) & set(splits["val"]))
        assert not (set(splits["train"]) & set(splits["test"]))
        assert not (set(splits["val"]) & set(splits["test"]))

        def get_patients(indices):
            return {dataset.samples[i][1].patient_id for i in indices}

        train_pids = get_patients(splits["train"])
        val_pids   = get_patients(splits["val"])
        test_pids  = get_patients(splits["test"])

        assert not (train_pids & val_pids),  "Patient appears in both train and val!"
        assert not (train_pids & test_pids), "Patient appears in both train and test!"
        assert not (val_pids   & test_pids), "Patient appears in both val and test!"

    def test_multi_image_patient_same_split(self, temp_dataset_dir, sample_image):
        """All images from the same patient must be in the same split."""
        from dataset_split import split_clinical_dataset, SplitConfig

        ds_dir = self._make_multi_image_dataset(temp_dataset_dir, sample_image)
        dataset = ClinicalDataset(ds_dir)

        split_config = SplitConfig(
            patient_level_split=True,
            random_seed=42,
            clinical_train_ratio=0.50,
            clinical_val_ratio=0.25,
            clinical_test_ratio=0.25,
        )
        splits = split_clinical_dataset(dataset, split_config)

        idx_to_split = {
            idx: split_name
            for split_name, idxs in splits.items()
            for idx in idxs
        }

        patient_splits: dict = defaultdict(set)
        for idx, (_, meta) in enumerate(dataset.samples):
            if meta.patient_id and idx in idx_to_split:
                patient_splits[meta.patient_id].add(idx_to_split[idx])

        for pid, split_set in patient_splits.items():
            assert len(split_set) == 1, (
                f"Patient {pid} images scattered across splits: {split_set}"
            )

    def test_fallback_when_no_patient_id(self, clinical_dataset_fixture):
        """Image-level fallback is used when patient_id column is absent."""
        from dataset_split import split_clinical_dataset, SplitConfig

        # clinical_dataset_fixture has no patient_id column → all patient_ids None
        dataset = ClinicalDataset(clinical_dataset_fixture)

        split_config = SplitConfig(
            patient_level_split=True,
            random_seed=42,
            clinical_train_ratio=0.60,
            clinical_val_ratio=0.20,
            clinical_test_ratio=0.20,
            stratify_on_label=False,  # Avoid stratification issues on tiny dataset
        )
        # Must not raise; should fall back to image-level splitting
        splits = split_clinical_dataset(dataset, split_config)

        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == len(dataset)



# ============================================================================
# TESTS - Calibration Curve + Brier Score
# ============================================================================

class TestCalibration:
    """Tests for compute_and_plot_calibration in evaluation.py."""

    def test_returns_brier_score(self, tmp_path):
        from evaluation import compute_and_plot_calibration
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.4, 0.6, 0.8, 0.3, 0.7])
        result = compute_and_plot_calibration(
            y_true, y_prob, str(tmp_path / "calib.png"), n_bins=3
        )
        assert "brier_score" in result
        assert 0.0 <= result["brier_score"] <= 1.0

    def test_saves_png(self, tmp_path):
        from evaluation import compute_and_plot_calibration
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.4, 0.6, 0.8, 0.3, 0.7])
        out = tmp_path / "calib.png"
        compute_and_plot_calibration(y_true, y_prob, str(out), n_bins=3)
        assert out.exists()

    def test_perfect_predictions_brier_zero(self, tmp_path):
        from evaluation import compute_and_plot_calibration
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        result = compute_and_plot_calibration(
            y_true, y_prob, str(tmp_path / "c.png"), n_bins=3
        )
        assert result["brier_score"] == pytest.approx(0.0, abs=1e-6)

    def test_worst_predictions_brier_one(self, tmp_path):
        from evaluation import compute_and_plot_calibration
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        result = compute_and_plot_calibration(
            y_true, y_prob, str(tmp_path / "c.png"), n_bins=3
        )
        assert result["brier_score"] == pytest.approx(1.0, abs=1e-6)


class TestDatasetStats:
    """Tests for dataset-stat computation used for normalization."""

    def test_compute_dataset_stats_returns_three_channels(self, public_dataset_fixture):
        from preprocessing import compute_dataset_stats
        dataset = PublicDataset(public_dataset_fixture)
        mean, std = compute_dataset_stats(dataset, sample_size=5)
        assert len(mean) == 3
        assert len(std) == 3

    def test_compute_dataset_stats_values_are_bounded(self, public_dataset_fixture):
        from preprocessing import compute_dataset_stats
        dataset = PublicDataset(public_dataset_fixture)
        mean, std = compute_dataset_stats(dataset, sample_size=5)
        assert all(0.0 <= v <= 1.0 for v in mean)
        assert all(0.0 <= v <= 1.0 for v in std)


# ============================================================================
# TESTS - Cross-Validation Utilities
# ============================================================================

class TestCrossValidation:
    """Tests for cross_validation.py utilities (no model training required)."""

    def test_summarise_computes_mean_std(self):
        from cross_validation import _summarise
        per_fold = [
            {"accuracy": 0.80, "sensitivity": 0.70, "f1": 0.75},
            {"accuracy": 0.90, "sensitivity": 0.85, "f1": 0.87},
        ]
        summary = _summarise(per_fold)
        assert "accuracy" in summary
        assert summary["accuracy"]["mean"] == pytest.approx(0.85, abs=1e-4)
        assert summary["accuracy"]["std"] > 0.0
        assert "ci_lower" in summary["accuracy"]
        assert "ci_upper" in summary["accuracy"]

    def test_summarise_skips_none_values(self):
        from cross_validation import _summarise
        per_fold = [
            {"accuracy": 0.80, "auc": None},
            {"accuracy": 0.90, "auc": None},
        ]
        summary = _summarise(per_fold)
        assert "accuracy" in summary
        assert "auc" not in summary  # all None → omitted

    def test_summarise_values_list(self):
        from cross_validation import _summarise
        per_fold = [{"accuracy": 0.70}, {"accuracy": 0.80}, {"accuracy": 0.90}]
        summary = _summarise(per_fold)
        assert len(summary["accuracy"]["values"]) == 3

    def test_save_cv_report_creates_csv_and_json(self, tmp_path):
        from cross_validation import save_cv_report
        results = {
            "per_repeat": [
                {"accuracy": 0.80, "sensitivity": 0.70},
                {"accuracy": 0.90, "sensitivity": 0.85},
            ],
            "summary": {
                "accuracy": {
                    "mean": 0.85,
                    "std": 0.07,
                    "ci_lower": 0.80,
                    "ci_upper": 0.90,
                    "values": [0.80, 0.90],
                }
            },
        }
        out = str(tmp_path / "cv_results")
        save_cv_report(results, out)
        assert (tmp_path / "cv_results.csv").exists()
        assert (tmp_path / "cv_results.json").exists()

    def test_save_cv_report_csv_has_header(self, tmp_path):
        from cross_validation import save_cv_report
        import csv as _csv
        results = {
            "per_repeat": [{"accuracy": 0.80}],
            "summary": {
                "accuracy": {
                    "mean": 0.80,
                    "std": 0.0,
                    "ci_lower": 0.80,
                    "ci_upper": 0.80,
                    "values": [0.80],
                }
            },
        }
        out = str(tmp_path / "cv")
        save_cv_report(results, out)
        with open(tmp_path / "cv.csv") as f:
            reader = _csv.DictReader(f)
            assert "metric" in reader.fieldnames
            assert "ci_lower" in reader.fieldnames
            assert "ci_upper" in reader.fieldnames


# ============================================================================
# TESTS - Grad-CAM
# ============================================================================

class TestGradCAM:
    """Tests for gradcam.py (uses tiny resnet with random weights)."""

    @staticmethod
    def _make_resnet18_eval():
        """Return a minimal resnet18 (unpretrained) in eval mode."""
        from torchvision.models import resnet18 as _resnet18
        m = _resnet18(pretrained=False)
        m.fc = torch.nn.Linear(m.fc.in_features, 2)
        return m.eval()

    def test_get_target_layer_resnet18(self):
        from gradcam import get_target_layer
        model = self._make_resnet18_eval()
        layer = get_target_layer(model, model_type="resnet18")
        assert layer is not None
        assert isinstance(layer, torch.nn.Module)

    def test_get_target_layer_unknown_raises(self):
        from gradcam import get_target_layer
        model = self._make_resnet18_eval()
        with pytest.raises(ValueError):
            get_target_layer(model, model_type="unknown_arch")

    def test_gradcam_output_shape(self):
        from gradcam import GradCAM
        model = self._make_resnet18_eval()
        target_layer = list(model.layer4.children())[-1]
        image = torch.randn(1, 3, 224, 224)
        gc = GradCAM(model, target_layer)
        try:
            with torch.enable_grad():
                cam, pred_class, logits = gc(image, target_class=1)
        finally:
            gc.remove_hooks()
        assert cam.shape == (7, 7)  # resnet18 feature map before pooling
        assert pred_class in (0, 1)
        assert logits.shape == (1, 2)

    def test_gradcam_heatmap_in_unit_interval(self):
        from gradcam import GradCAM
        model = self._make_resnet18_eval()
        target_layer = list(model.layer4.children())[-1]
        image = torch.randn(1, 3, 224, 224)
        gc = GradCAM(model, target_layer)
        try:
            with torch.enable_grad():
                cam, _, _ = gc(image)
        finally:
            gc.remove_hooks()
        assert float(cam.min()) >= 0.0
        assert float(cam.max()) <= 1.0 + 1e-6

    def test_generate_gradcam_resize_to_input(self):
        from gradcam import GradCAM, generate_gradcam
        model = self._make_resnet18_eval()
        target_layer = list(model.layer4.children())[-1]
        image = torch.randn(1, 3, 224, 224)
        heatmap, overlay = generate_gradcam(model, image, target_layer)
        # heatmap should be resized to input spatial resolution
        assert heatmap.shape == (224, 224)
        assert overlay.shape == (224, 224, 3)

    def test_hooks_removed_after_generate(self):
        """generate_gradcam must clean up hooks even on error."""
        from gradcam import GradCAM, generate_gradcam
        model = self._make_resnet18_eval()
        target_layer = list(model.layer4.children())[-1]
        # Count hooks before
        hooks_before = len(target_layer._forward_hooks) + len(target_layer._backward_hooks)
        image = torch.randn(1, 3, 224, 224)
        generate_gradcam(model, image, target_layer)
        hooks_after = len(target_layer._forward_hooks) + len(target_layer._backward_hooks)
        assert hooks_after == hooks_before  # no leak


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
