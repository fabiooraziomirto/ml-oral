"""
Cross-Validation Module — Repeated Holdout and Stratified K-Fold
================================================================
Provides robust performance estimation for small clinical datasets.

Both strategies honour patient-level splitting (via SplitConfig.patient_level_split)
so that images from the same patient never appear in training and test simultaneously.

Usage
-----
    from cross_validation import run_repeated_holdout, run_kfold, save_cv_report

    results = run_repeated_holdout(
        dataset=clinical_ds,
        train_eval_fn=my_train_eval_fn,   # Callable[[Dict[str, DataLoader]], Dict[str, float]]
        n_repeats=5,
    )
    save_cv_report(results, output_path="cv_results")
    print(results["summary"])
"""

import json
import csv
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

from data_loader import ClinicalDataset
from dataset_split import SplitConfig, split_clinical_dataset
from preprocessing import AugmentationConfig, TransformPipeline, TransformedDataset

logger = logging.getLogger(__name__)

# Metrics reported in every fold
_METRIC_KEYS = ["accuracy", "sensitivity", "specificity", "auc", "f1", "kappa"]


# ============================================================================
# DATALOADER FACTORY (lightweight, avoids importing DataLoaderFactory)
# ============================================================================

def _make_loaders(
    dataset: ClinicalDataset,
    splits: Dict[str, List[int]],
    batch_size: int,
    num_workers: int,
) -> Dict[str, DataLoader]:
    """Build train/val/test DataLoaders from pre-computed index splits."""
    pipeline = TransformPipeline(augmentation_config=AugmentationConfig())
    tf = pipeline.get_transforms_dict()

    loaders: Dict[str, DataLoader] = {}
    for split_name, indices in splits.items():
        transform = tf["train"] if split_name == "train" else tf[split_name]
        td = TransformedDataset(Subset(dataset, indices), transform)
        loaders[split_name] = DataLoader(
            td,
            batch_size=batch_size,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            pin_memory=False,
        )
    return loaders


# ============================================================================
# PATIENT-LEVEL K-FOLD SPLITTER
# ============================================================================

def _patient_kfold_splits(
    dataset: ClinicalDataset,
    k: int,
    seed: int,
    stratify: bool,
) -> List[Dict[str, List[int]]]:
    """
    Build k stratified folds at the patient level.

    Each fold returns a dict with 'train' and 'test' image-index lists.
    Patients are never split across train/test within a fold.

    Falls back to image-level folding if patient_id is not populated.
    """
    from collections import defaultdict as _dd

    patient_to_indices: Dict[str, List[int]] = _dd(list)
    patient_to_label: Dict[str, int] = {}

    for idx, (_, meta) in enumerate(dataset.samples):
        pid = meta.patient_id or f"__img_{idx}__"
        patient_to_indices[pid].append(idx)
        if pid not in patient_to_label:
            patient_to_label[pid] = int(meta.label)

    unique_patients = np.array(list(patient_to_indices.keys()))
    patient_labels  = np.array([patient_to_label[p] for p in unique_patients])

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    stratify_arr = patient_labels if stratify else np.zeros(len(unique_patients), dtype=int)

    folds: List[Dict[str, List[int]]] = []
    for train_p_idx, test_p_idx in skf.split(unique_patients, stratify_arr):
        train_images = [
            i for p in unique_patients[train_p_idx]
            for i in patient_to_indices[p]
        ]
        test_images = [
            i for p in unique_patients[test_p_idx]
            for i in patient_to_indices[p]
        ]
        folds.append({"train": train_images, "test": test_images})

    return folds


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def _bootstrap_confidence_interval(
    values: List[float],
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Estimate a percentile-bootstrap confidence interval for the mean."""
    if not values:
        raise ValueError("Cannot bootstrap an empty value list")

    arr = np.array(values, dtype=float)
    if len(arr) == 1:
        point = round(float(arr[0]), 4)
        return {"ci_lower": point, "ci_upper": point}

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_resamples, dtype=float)
    for idx in range(n_resamples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means[idx] = sample.mean()

    alpha = 1.0 - confidence_level
    return {
        "ci_lower": round(float(np.quantile(boot_means, alpha / 2.0)), 4),
        "ci_upper": round(float(np.quantile(boot_means, 1.0 - alpha / 2.0)), 4),
    }


def _summarise(per_repeat: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Compute mean, std, and 95% bootstrap CI for each metric across repeats/folds.

    Returns a dict of {metric: {mean, std, ci_lower, ci_upper, values}}.
    """
    summary: Dict[str, Dict] = {}
    for key in _METRIC_KEYS:
        values = [r.get(key) for r in per_repeat if r.get(key) is not None]
        if values:
            ci = _bootstrap_confidence_interval(values)
            summary[key] = {
                "mean": round(float(np.mean(values)), 4),
                "std": round(float(np.std(values, ddof=1)), 4) if len(values) > 1 else 0.0,
                "ci_lower": ci["ci_lower"],
                "ci_upper": ci["ci_upper"],
                "values": [round(v, 4) for v in values],
            }
    return summary


# ============================================================================
# REPEATED HOLDOUT
# ============================================================================

def run_repeated_holdout(
    dataset: ClinicalDataset,
    train_eval_fn: Callable[[Dict[str, DataLoader]], Dict[str, float]],
    n_repeats: int = 5,
    seeds: Optional[List[int]] = None,
    split_config_base: Optional[SplitConfig] = None,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Dict:
    """
    Run the training+evaluation pipeline ``n_repeats`` times with different seeds.

    Patient-level splitting is used when ``split_config_base.patient_level_split``
    is True (default) and patient_id fields are populated.

    Parameters
    ----------
    dataset          : ClinicalDataset to split and evaluate
    train_eval_fn    : callable that receives a ``Dict[str, DataLoader]``
                       (keys: 'train', 'val', 'test') and returns a metrics dict
                       with keys from {accuracy, sensitivity, specificity, auc, f1, kappa}
    n_repeats        : number of independent repetitions
    seeds            : list of seeds; auto-generated if None
    split_config_base: base SplitConfig (seed overridden per repeat)
    batch_size       : batch size for DataLoaders passed to train_eval_fn
    num_workers      : DataLoader workers

    Returns
    -------
    dict with:
        "per_repeat" : List[Dict] — metrics for each repeat
        "summary"    : Dict       — {metric: {mean, std, values}}
    """
    if seeds is None:
        rng = np.random.default_rng(42)
        seeds = rng.integers(0, 2**31, size=n_repeats).tolist()

    if split_config_base is None:
        split_config_base = SplitConfig()

    per_repeat: List[Dict[str, float]] = []

    for repeat_idx, seed in enumerate(seeds[:n_repeats]):
        logger.info(f"Repeated holdout {repeat_idx + 1}/{n_repeats}  (seed={seed})")
        print(f"\n[CV] Repeat {repeat_idx + 1}/{n_repeats}  seed={seed}")

        # Build split with this seed
        cfg = SplitConfig(
            clinical_train_ratio=split_config_base.clinical_train_ratio,
            clinical_val_ratio=split_config_base.clinical_val_ratio,
            clinical_test_ratio=split_config_base.clinical_test_ratio,
            stratify_on_label=split_config_base.stratify_on_label,
            patient_level_split=split_config_base.patient_level_split,
            random_seed=int(seed),
        )
        splits = split_clinical_dataset(dataset, cfg)
        loaders = _make_loaders(dataset, splits, batch_size, num_workers)

        metrics = train_eval_fn(loaders)
        per_repeat.append(metrics)
        print(f"  → {', '.join(f'{k}={v:.4f}' for k, v in metrics.items() if v is not None)}")

    summary = _summarise(per_repeat)
    logger.info("Repeated holdout complete")
    return {"per_repeat": per_repeat, "summary": summary}


# ============================================================================
# STRATIFIED K-FOLD
# ============================================================================

def run_kfold(
    dataset: ClinicalDataset,
    train_eval_fn: Callable[[Dict[str, DataLoader]], Dict[str, float]],
    k: int = 5,
    seed: int = 42,
    split_config_base: Optional[SplitConfig] = None,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Dict:
    """
    Run stratified k-fold cross-validation at the patient level.

    K-fold is recommended for datasets with > 200 patients.  For smaller
    datasets use ``run_repeated_holdout`` which gives a proper held-out
    test set each time.

    Parameters
    ----------
    dataset          : ClinicalDataset to fold
    train_eval_fn    : callable — same signature as for run_repeated_holdout;
                       receives loaders with keys 'train' and 'test' only
    k                : number of folds
    seed             : random seed for fold generation
    split_config_base: provides stratify_on_label and patient_level_split flags
    batch_size       : DataLoader batch size
    num_workers      : DataLoader workers

    Returns
    -------
    dict with:
        "per_fold" : List[Dict] — metrics for each fold
        "summary"  : Dict       — {metric: {mean, std, values}}
    """
    if split_config_base is None:
        split_config_base = SplitConfig()

    folds = _patient_kfold_splits(
        dataset,
        k=k,
        seed=seed,
        stratify=split_config_base.stratify_on_label,
    )

    per_fold: List[Dict[str, float]] = []

    for fold_idx, fold_splits in enumerate(folds):
        logger.info(f"K-fold {fold_idx + 1}/{k}")
        print(f"\n[CV] Fold {fold_idx + 1}/{k}"
              f"  train={len(fold_splits['train'])}  test={len(fold_splits['test'])}")

        loaders = _make_loaders(dataset, fold_splits, batch_size, num_workers)
        metrics = train_eval_fn(loaders)
        per_fold.append(metrics)
        print(f"  → {', '.join(f'{k_}={v:.4f}' for k_, v in metrics.items() if v is not None)}")

    summary = _summarise(per_fold)
    logger.info("K-fold complete")
    return {"per_fold": per_fold, "summary": summary}


# ============================================================================
# REPORT OUTPUT
# ============================================================================

def save_cv_report(results: Dict, output_path: str) -> None:
    """
    Save cross-validation results to a CSV summary and a JSON file.

    Parameters
    ----------
    results     : dict returned by run_repeated_holdout or run_kfold
    output_path : path prefix (without extension); '.csv' and '.json' appended
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Determine per-fold/repeat list key
    rows = results.get("per_repeat") or results.get("per_fold", [])
    summary = results.get("summary", {})

    # ── CSV summary ──────────────────────────────────────────────────────────
    csv_path = out.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["metric", "mean", "std", "ci_lower", "ci_upper", "values"],
        )
        writer.writeheader()
        for metric, stats in summary.items():
            writer.writerow({
                "metric": metric,
                "mean": stats.get("mean", ""),
                "std": stats.get("std", ""),
                "ci_lower": stats.get("ci_lower", ""),
                "ci_upper": stats.get("ci_upper", ""),
                "values": ";".join(str(v) for v in stats.get("values", [])),
            })
    print(f"  CV results CSV: {csv_path}")

    # ── JSON (full results + summary) ────────────────────────────────────────
    json_path = out.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  CV results JSON: {json_path}")

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n  {'Metric':<15}  {'Mean':>8}  {'±Std':>8}  {'95% CI':>19}")
    print(f"  {'-'*56}")
    for metric, stats in summary.items():
        print(
            f"  {metric:<15}  {stats['mean']:>8.4f}  {stats['std']:>8.4f}  "
            f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]"
        )
