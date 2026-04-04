"""
Clinical AI Error Analysis Module

Analyses disagreements between AI predictions, clinician diagnoses,
and biopsy ground truth. Produces a structured DataFrame and CSV with
per-case categorisation, plus ranked subsets of the most clinically
interesting cases for expert review.

Integrates directly with evaluation.run_inference() output.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ErrorAnalysisConfig:
    """Configuration for the error analysis pipeline."""
    output_dir: str = "./error_analysis"
    all_cases_filename: str = "error_analysis.csv"
    top_cases_filename: str = "top_cases_for_review.csv"
    top_n: int = 20                  # Number of top cases to surface for review
    positive_label: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    threshold: float = 0.5           # Decision threshold


# ============================================================================
# CASE CATEGORIES
# ============================================================================

# All mutually exclusive case categories
CATEGORY_AI_CORRECT_CLINICIAN_WRONG  = "AI correct, clinician wrong"
CATEGORY_CLINICIAN_CORRECT_AI_WRONG  = "Clinician correct, AI wrong"
CATEGORY_BOTH_WRONG                  = "Both wrong"
CATEGORY_BOTH_CORRECT                = "Both correct"
#   Sub-category for cases where clinician label is unavailable
CATEGORY_AI_CORRECT_NO_CLINICIAN     = "AI correct (no clinician label)"
CATEGORY_AI_WRONG_NO_CLINICIAN       = "AI wrong (no clinician label)"


def _assign_category(
    biopsy: int,
    ai_pred: int,
    clinician: Optional[int],
) -> str:
    """Return the disagreement category for a single case."""
    ai_correct = (ai_pred == biopsy)

    if clinician is None:
        return (
            CATEGORY_AI_CORRECT_NO_CLINICIAN
            if ai_correct
            else CATEGORY_AI_WRONG_NO_CLINICIAN
        )

    clinician_correct = (clinician == biopsy)

    if ai_correct and clinician_correct:
        return CATEGORY_BOTH_CORRECT
    if ai_correct and not clinician_correct:
        return CATEGORY_AI_CORRECT_CLINICIAN_WRONG
    if not ai_correct and clinician_correct:
        return CATEGORY_CLINICIAN_CORRECT_AI_WRONG
    return CATEGORY_BOTH_WRONG  # both wrong


# ============================================================================
# INFERENCE  (thin wrapper so this module is self-contained)
# ============================================================================

def run_inference(
    model: nn.Module,
    dataloader: DataLoader,
    config: ErrorAnalysisConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """
    Run model inference and collect per-sample outputs and metadata.

    Returns
    -------
    probs          : (N,) positive-class probabilities
    ai_preds       : (N,) binary predictions
    biopsy_labels  : (N,) gold-standard labels
    metadata_list  : per-sample metadata dicts
    """
    model.eval()
    model.to(config.device)

    all_probs, all_preds, all_biopsy, all_meta = [], [], [], []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                images, labels, metadata = batch
            else:
                images, labels = batch
                metadata = [{} for _ in range(len(labels))]

            images = images.to(config.device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, config.positive_label].cpu().numpy()
            preds = (probs >= config.threshold).astype(int)

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_biopsy.extend(labels.cpu().numpy().tolist())
            all_meta.extend(
                [m if isinstance(m, dict) else {} for m in metadata]
            )

    return (
        np.array(all_probs),
        np.array(all_preds),
        np.array(all_biopsy),
        all_meta,
    )


# ============================================================================
# DATAFRAME CONSTRUCTION
# ============================================================================

def build_error_dataframe(
    probs: np.ndarray,
    ai_preds: np.ndarray,
    biopsy_labels: np.ndarray,
    metadata_list: List[Dict],
) -> pd.DataFrame:
    """
    Build a per-case structured DataFrame with error categorisation.

    Columns
    -------
    image_name        : str
    image_id          : str  (if available)
    biopsy_diagnosis  : int  (0 / 1) — gold standard
    clinician_diagnosis: int or NaN
    ai_prediction     : int  (0 / 1)
    ai_probability    : float  — P(positive)
    ai_correct        : bool
    clinician_correct : bool or NaN
    category          : str  — one of the CATEGORY_* constants
    lesion_type       : str or NaN
    location          : str or NaN
    dataset_source    : str or NaN
    """
    records = []

    for i, meta in enumerate(metadata_list):
        biopsy = int(biopsy_labels[i])
        ai_pred = int(ai_preds[i])
        prob = float(probs[i])

        clinician_raw = meta.get("clinician_diagnosis")
        clinician = int(clinician_raw) if clinician_raw is not None else None

        category = _assign_category(biopsy, ai_pred, clinician)

        ai_correct = (ai_pred == biopsy)
        clinician_correct = (clinician == biopsy) if clinician is not None else None

        records.append({
            "image_name":          meta.get("image_name", f"sample_{i}"),
            "image_id":            meta.get("image_id", ""),
            "biopsy_diagnosis":    biopsy,
            "clinician_diagnosis": clinician,       # NaN when absent (pandas handles None → NaN)
            "ai_prediction":       ai_pred,
            "ai_probability":      round(prob, 4),
            "ai_correct":          ai_correct,
            "clinician_correct":   clinician_correct,
            "category":            category,
            "lesion_type":         meta.get("lesion_type"),
            "location":            meta.get("location"),
            "dataset_source":      meta.get("dataset_source"),
        })

    df = pd.DataFrame(records)
    return df


# ============================================================================
# CATEGORY SUMMARY
# ============================================================================

def summarise_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary table with counts and percentages per category.

    Returns
    -------
    DataFrame with columns: category, count, percentage
    """
    total = len(df)
    summary = (
        df.groupby("category", observed=True)
        .size()
        .reset_index(name="count")
    )
    summary["percentage"] = (summary["count"] / total * 100).round(2)
    summary = summary.sort_values("count", ascending=False).reset_index(drop=True)
    return summary


# ============================================================================
# TOP-N CASE SELECTION
# ============================================================================

def _confidence_gap(df: pd.DataFrame) -> pd.Series:
    """
    'Confidence gap' for ranking: how far the AI probability is from 0.5.
    Higher = more confidently wrong → more educationally interesting.
    """
    return (df["ai_probability"] - 0.5).abs()


def select_top_cases(
    df: pd.DataFrame,
    top_n: int = 20,
    focus_categories: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Select the top-N most interesting cases for expert review.

    Strategy
    --------
    Among error cases, rank by AI confidence gap (most confident errors
    are most insightful for human reviewers). If focus_categories is given,
    restrict to those categories; otherwise include all non-'Both correct'
    categories, with priority:
      1. AI correct, clinician wrong  — AI adds value
      2. Clinician correct, AI wrong  — AI fails where humans succeed
      3. Both wrong                   — hardest cases

    Parameters
    ----------
    df               : full error analysis DataFrame from build_error_dataframe()
    top_n            : maximum number of cases to return
    focus_categories : restrict to these categories (None = all error types)

    Returns
    -------
    Subset DataFrame (≤ top_n rows), sorted by priority and confidence gap.
    """
    priority_order = {
        CATEGORY_AI_CORRECT_CLINICIAN_WRONG: 0,
        CATEGORY_CLINICIAN_CORRECT_AI_WRONG: 1,
        CATEGORY_BOTH_WRONG:                 2,
        CATEGORY_AI_CORRECT_NO_CLINICIAN:    3,
        CATEGORY_AI_WRONG_NO_CLINICIAN:      4,
        CATEGORY_BOTH_CORRECT:               5,
    }

    if focus_categories is not None:
        mask = df["category"].isin(focus_categories)
    else:
        # Exclude 'both correct' by default — those are not interesting for error analysis
        mask = df["category"] != CATEGORY_BOTH_CORRECT

    subset = df[mask].copy()
    subset["_priority"]   = subset["category"].map(priority_order).fillna(99)
    subset["_conf_gap"]   = _confidence_gap(subset)

    top = (
        subset
        .sort_values(["_priority", "_conf_gap"], ascending=[True, False])
        .head(top_n)
        .drop(columns=["_priority", "_conf_gap"])
        .reset_index(drop=True)
    )
    return top


# ============================================================================
# SAVE OUTPUT
# ============================================================================

def save_csv(df: pd.DataFrame, output_path: str) -> None:
    """Save DataFrame to CSV, creating parent directories as needed."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}  ({len(df)} rows)")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def analyse_errors(
    model: nn.Module,
    test_dataloader: DataLoader,
    config: Optional[ErrorAnalysisConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full error analysis pipeline.

    Parameters
    ----------
    model            : trained PyTorch model
    test_dataloader  : DataLoader over the clinical test set;
                       batches must be (images, labels, metadata_dicts)
    config           : ErrorAnalysisConfig; uses defaults if None

    Returns
    -------
    df_all    : complete per-case DataFrame with error categories
    df_top    : top-N cases recommended for expert review
    """
    if config is None:
        config = ErrorAnalysisConfig()

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("CLINICAL AI ERROR ANALYSIS")
    print("=" * 70)

    # ── 1. Inference ────────────────────────────────────────────────────────
    print("\n[1/4] Running inference...")
    probs, ai_preds, biopsy_labels, metadata_list = run_inference(
        model, test_dataloader, config
    )
    n = len(biopsy_labels)
    print(f"  Samples evaluated : {n}")

    has_clinician = sum(
        1 for m in metadata_list
        if isinstance(m, dict) and m.get("clinician_diagnosis") is not None
    )
    print(f"  With clinician label: {has_clinician} / {n}")

    # ── 2. Build DataFrame ──────────────────────────────────────────────────
    print("\n[2/4] Categorising cases...")
    df_all = build_error_dataframe(probs, ai_preds, biopsy_labels, metadata_list)

    summary = summarise_categories(df_all)
    print("\n  Category breakdown:")
    print(
        summary.to_string(
            index=False,
            columns=["category", "count", "percentage"],
        )
    )

    # ── 3. Top-N cases ──────────────────────────────────────────────────────
    print(f"\n[3/4] Selecting top-{config.top_n} cases for review...")
    df_top = select_top_cases(df_all, top_n=config.top_n)
    print(f"  Top cases selected: {len(df_top)}")
    if not df_top.empty:
        print(f"  Category distribution in top cases:")
        top_summary = df_top["category"].value_counts().reset_index()
        top_summary.columns = ["category", "count"]
        for _, row in top_summary.iterrows():
            print(f"    {row['category']}: {row['count']}")

    # ── 4. Save CSVs ────────────────────────────────────────────────────────
    print("\n[4/4] Saving results...")
    all_path = os.path.join(config.output_dir, config.all_cases_filename)
    top_path = os.path.join(config.output_dir, config.top_cases_filename)
    save_csv(df_all, all_path)
    save_csv(df_top, top_path)

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"  Output directory : {os.path.abspath(config.output_dir)}")
    print(f"  All cases CSV    : {all_path}")
    print(f"  Top cases CSV    : {top_path}")

    return df_all, df_top


# ============================================================================
# CONVENIENCE: analyse from pre-computed arrays (no model/dataloader needed)
# ============================================================================

def analyse_errors_from_arrays(
    image_names: List[str],
    ai_predictions: List[int],
    ai_probabilities: List[float],
    biopsy_labels: List[int],
    clinician_labels: Optional[List[Optional[int]]] = None,
    extra_metadata: Optional[List[Dict]] = None,
    config: Optional[ErrorAnalysisConfig] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run error analysis directly from pre-computed prediction arrays,
    without needing a model or DataLoader.

    Useful when predictions have already been collected (e.g., from
    evaluation.run_inference()).

    Parameters
    ----------
    image_names      : file names, one per sample
    ai_predictions   : binary AI predictions (0/1)
    ai_probabilities : positive-class probabilities
    biopsy_labels    : gold-standard biopsy labels (0/1)
    clinician_labels : clinician binary predictions; None entries are allowed
    extra_metadata   : optional list of dicts with 'lesion_type', 'location', etc.
    config           : ErrorAnalysisConfig

    Returns
    -------
    df_all, df_top
    """
    if config is None:
        config = ErrorAnalysisConfig()

    n = len(image_names)
    clinician_labels = clinician_labels or [None] * n
    extra_metadata   = extra_metadata   or [{} for _ in range(n)]

    metadata_list = []
    for i in range(n):
        meta = {
            "image_name":          image_names[i],
            "clinician_diagnosis": clinician_labels[i],
        }
        meta.update(extra_metadata[i])
        metadata_list.append(meta)

    probs      = np.array(ai_probabilities)
    ai_preds   = np.array(ai_predictions)
    biopsy_arr = np.array(biopsy_labels)

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    df_all = build_error_dataframe(probs, ai_preds, biopsy_arr, metadata_list)
    df_top = select_top_cases(df_all, top_n=config.top_n)

    all_path = os.path.join(config.output_dir, config.all_cases_filename)
    top_path = os.path.join(config.output_dir, config.top_cases_filename)
    save_csv(df_all, all_path)
    save_csv(df_top, top_path)

    return df_all, df_top


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example — integrate with evaluation.py and dataset_split.py:

        from training_pipeline import BinaryClassificationModel
        from dataset_split import get_clinical_dataloaders
        from data_loader import get_clinical_dataset
        from error_analysis import analyse_errors, ErrorAnalysisConfig
        import torch

        model = BinaryClassificationModel(model_type="resnet18")
        ckpt = torch.load("checkpoints/checkpoint_clinical.pth", map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])

        dataset = get_clinical_dataset("data/clinical_dataset")
        loaders = get_clinical_dataloaders(dataset, batch_size=32)

        config = ErrorAnalysisConfig(output_dir="./error_analysis", top_n=20)
        df_all, df_top = analyse_errors(model, loaders["test"], config)

    Or from pre-computed arrays (e.g., reusing evaluation.run_inference output):

        from error_analysis import analyse_errors_from_arrays

        df_all, df_top = analyse_errors_from_arrays(
            image_names      = ["img_001.jpg", "img_002.jpg", ...],
            ai_predictions   = [1, 0, ...],
            ai_probabilities = [0.91, 0.23, ...],
            biopsy_labels    = [1, 1, ...],
            clinician_labels = [0, 1, ...],
        )
    """
    print("Error analysis module loaded. See __main__ docstring for usage.")
