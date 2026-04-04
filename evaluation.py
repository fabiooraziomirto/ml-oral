"""
Clinical AI Evaluation Module

Full evaluation of a trained medical imaging model against:
- Biopsy (gold standard)
- Clinician diagnosis

Produces publication-ready metrics, confusion matrices, ROC curves,
and structured reports (dict + CSV).
"""

import os
import csv
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    calibration_curve,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class EvaluationConfig:
    """Configuration for the evaluation pipeline."""
    output_dir: str = "./evaluation_results"
    roc_curve_filename: str = "roc_curve.png"
    report_csv_filename: str = "metrics_report.csv"
    confusion_matrix_filename: str = "confusion_matrices.png"
    positive_label: int = 1           # Class index treated as "positive"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    threshold: float = 0.5            # Decision threshold for binary classification
    find_best_threshold: bool = False  # If True, also report the Youden-optimal threshold
    predictions_csv_filename: str = "predictions.csv"
    calibration_filename: str = "calibration_curve.png"


# ============================================================================
# INFERENCE
# ============================================================================

def run_inference(
    model: nn.Module,
    dataloader: DataLoader,
    config: EvaluationConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
    """
    Run model inference on a dataloader.

    Returns
    -------
    probabilities      : shape (N,)  — P(positive) for each sample
    predicted_labels   : shape (N,)  — argmax predictions
    biopsy_labels      : shape (N,)  — gold-standard biopsy label
    metadata_list      : list of per-sample metadata dicts
    """
    model.eval()
    model.to(config.device)

    all_probs: List[float] = []
    all_preds: List[int] = []
    all_biopsy: List[int] = []
    all_meta: List[Dict] = []

    with torch.no_grad():
        for batch in dataloader:
            # Support both (images, labels, metadata) and (images, labels) tuples
            if len(batch) == 3:
                images, labels, metadata = batch
            else:
                images, labels = batch
                metadata = [{} for _ in range(len(labels))]

            images = images.to(config.device)
            logits = model(images)                              # (B, 2)
            probs = torch.softmax(logits, dim=1)[:, config.positive_label]

            preds = (probs.cpu().numpy() >= config.threshold).astype(int)

            all_probs.extend(probs.cpu().numpy().tolist())
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
# METRIC COMPUTATION
# ============================================================================

def _safe_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Specificity = TN / (TN + FP)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn = cm[0, 0]
    fp = cm[0, 1]
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def _safe_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    """Cohen's Kappa — returns None when it cannot be computed."""
    try:
        return float(cohen_kappa_score(y_true, y_pred))
    except Exception:
        return None


def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    label: str = "",
) -> Dict[str, Optional[float]]:
    """
    Compute a full binary-classification metric set.

    Parameters
    ----------
    y_true  : ground-truth labels
    y_pred  : predicted labels
    y_prob  : predicted probabilities for the positive class (for ROC-AUC)
    label   : descriptive label used in printed output

    Returns
    -------
    metrics dict with keys:
        accuracy, sensitivity, specificity, precision, f1, roc_auc, kappa
    """
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    spec = _safe_specificity(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    kappa = _safe_kappa(y_true, y_pred)

    roc_auc: Optional[float] = None
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass

    metrics = {
        "accuracy": round(acc, 4),
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "precision": round(prec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
        "kappa": round(kappa, 4) if kappa is not None else None,
    }

    if label:
        print(f"\n  [{label}]")
        for k, v in metrics.items():
            print(f"    {k:<15}: {v}")

    return metrics


# ============================================================================
# CONFUSION MATRIX
# ============================================================================

def compute_and_plot_confusion_matrices(
    ai_vs_biopsy: Tuple[np.ndarray, np.ndarray],
    clinician_vs_biopsy: Optional[Tuple[np.ndarray, np.ndarray]],
    ai_vs_clinician: Optional[Tuple[np.ndarray, np.ndarray]],
    output_path: str,
) -> Dict[str, np.ndarray]:
    """
    Compute and save confusion matrices as a combined figure.

    Parameters
    ----------
    ai_vs_biopsy        : (y_true_biopsy, y_pred_ai)
    clinician_vs_biopsy : (y_true_biopsy, y_pred_clinician) or None
    ai_vs_clinician     : (y_true_clinician, y_pred_ai) or None
    output_path         : file path for the saved figure

    Returns
    -------
    dict of {name: ndarray} confusion matrices
    """
    panels = [("AI vs Biopsy", ai_vs_biopsy)]
    if clinician_vs_biopsy is not None:
        panels.append(("Clinician vs Biopsy", clinician_vs_biopsy))
    if ai_vs_clinician is not None:
        panels.append(("AI vs Clinician", ai_vs_clinician))

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    cms: Dict[str, np.ndarray] = {}

    for ax, (title, (y_true, y_pred)) in zip(axes, panels):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cms[title] = cm

        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted label", fontsize=11)
        ax.set_ylabel("True label", fontsize=11)
        tick_marks = [0, 1]
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(["Negative", "Positive"], fontsize=10)
        ax.set_yticklabels(["Negative", "Positive"], fontsize=10)

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j, i, format(cm[i, j], "d"),
                    ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > thresh else "black",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confusion matrices saved: {output_path}")
    return cms


# ============================================================================
# ROC CURVE
# ============================================================================

def plot_roc_curve(
    y_true_biopsy: np.ndarray,
    y_prob_ai: np.ndarray,
    y_pred_clinician: Optional[np.ndarray],
    output_path: str,
) -> None:
    """
    Plot and save the ROC curve (AI) with optional clinician operating point.

    Parameters
    ----------
    y_true_biopsy   : gold-standard biopsy labels
    y_prob_ai       : AI positive-class probabilities
    y_pred_clinician: clinician binary predictions (plotted as a point if provided)
    output_path     : file path for the saved figure
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    # AI ROC curve
    fpr, tpr, _ = roc_curve(y_true_biopsy, y_prob_ai)
    auc_score = roc_auc_score(y_true_biopsy, y_prob_ai)
    ax.plot(fpr, tpr, lw=2, color="#2563EB",
            label=f"AI model (AUC = {auc_score:.3f})")

    # Clinician operating point
    if y_pred_clinician is not None:
        cl_sens = recall_score(y_true_biopsy, y_pred_clinician,
                               pos_label=1, zero_division=0)
        cl_spec = _safe_specificity(y_true_biopsy, y_pred_clinician)
        ax.scatter(1 - cl_spec, cl_sens, s=120, zorder=5, color="#DC2626",
                   label=f"Clinician (Sens={cl_sens:.3f}, Spec={cl_spec:.3f})")

    # Reference diagonal
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")

    ax.set_xlabel("1 – Specificity (FPR)", fontsize=12)
    ax.set_ylabel("Sensitivity (TPR)", fontsize=12)
    ax.set_title("ROC Curve — AI vs Biopsy (Gold Standard)", fontsize=13,
                 fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.grid(alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ROC curve saved: {output_path}")


# ============================================================================
# REPORT GENERATION
# ============================================================================

def build_metrics_report(
    ai_vs_biopsy_metrics: Dict,
    clinician_vs_biopsy_metrics: Optional[Dict],
    ai_vs_clinician_metrics: Optional[Dict],
) -> Dict[str, Dict]:
    """
    Combine per-comparison metric dicts into a single structured report.
    """
    report = {"AI vs Biopsy": ai_vs_biopsy_metrics}
    if clinician_vs_biopsy_metrics is not None:
        report["Clinician vs Biopsy"] = clinician_vs_biopsy_metrics
    if ai_vs_clinician_metrics is not None:
        report["AI vs Clinician"] = ai_vs_clinician_metrics
    return report


def save_report_csv(report: Dict[str, Dict], output_path: str) -> None:
    """
    Write the metrics report to a CSV file.

    Columns: comparison, accuracy, sensitivity, specificity,
             precision, f1, roc_auc, kappa
    """
    fieldnames = [
        "comparison", "accuracy", "sensitivity", "specificity",
        "precision", "f1", "roc_auc", "kappa", "brier_score",
    ]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for comparison, metrics in report.items():
            row = {"comparison": comparison}
            row.update({k: metrics.get(k, "") for k in fieldnames[1:]})
            writer.writerow(row)

    print(f"  Metrics CSV saved: {output_path}")


def save_report_json(report: Dict[str, Dict], output_path: str) -> None:
    """Save the metrics report as a JSON file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Metrics JSON saved: {output_path}")


# ============================================================================
# THRESHOLD ANALYSIS
# ============================================================================

def compute_threshold_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict:
    """
    Scan decision thresholds and identify the Youden-optimal operating point.

    Youden's J = Sensitivity + Specificity - 1.  The threshold that maximises J
    balances sensitivity and specificity without external prior on class costs.

    Returns a dict with ``best_threshold``, ``best_youden_j``, the corresponding
    ``best_sensitivity`` / ``best_specificity``, and a ``threshold_table`` list
    covering clinically relevant thresholds (0.3, 0.4, 0.5, 0.6, 0.7, and the
    Youden optimum).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificities = 1.0 - fpr
    youden_j = tpr + specificities - 1.0

    best_idx = int(np.argmax(youden_j))
    best_thr = float(thresholds[best_idx])

    probe = sorted({0.3, 0.4, 0.5, 0.6, 0.7, round(best_thr, 3)})
    table = []
    for t in probe:
        preds = (y_prob >= t).astype(int)
        table.append({
            "threshold":   round(t, 3),
            "sensitivity": round(float(recall_score(y_true, preds, pos_label=1, zero_division=0)), 4),
            "specificity": round(float(_safe_specificity(y_true, preds)), 4),
            "precision":   round(float(precision_score(y_true, preds, pos_label=1, zero_division=0)), 4),
            "f1":          round(float(f1_score(y_true, preds, pos_label=1, zero_division=0)), 4),
        })

    return {
        "best_threshold":   round(best_thr, 3),
        "best_youden_j":    round(float(youden_j[best_idx]), 4),
        "best_sensitivity": round(float(tpr[best_idx]), 4),
        "best_specificity": round(float(specificities[best_idx]), 4),
        "threshold_table":  table,
    }


# ============================================================================
# CALIBRATION INSIGHT
# ============================================================================

def compute_calibration_insight(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 5,
) -> Dict:
    """
    Lightweight reliability check via equal-width probability bins.

    Compares mean predicted probability to the observed positive rate in each bin.
    A well-calibrated model has mean_prob approximately equal to observed_rate per bin.
    A positive gap means the model is overconfident; negative means underconfident.

    Returns a dict with an overall ``calibration_summary`` string and a
    ``bin_reliability`` list for per-bin inspection.
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_data = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if not mask.any():
            continue
        mean_prob = float(y_prob[mask].mean())
        obs_rate  = float(y_true[mask].mean())
        bin_data.append({
            "bin":           f"{lo:.1f}-{hi:.1f}",
            "n":             int(mask.sum()),
            "mean_prob":     round(mean_prob, 3),
            "observed_rate": round(obs_rate, 3),
            "gap":           round(mean_prob - obs_rate, 3),
        })

    mean_pred = float(y_prob.mean())
    frac_pos  = float(y_true.mean())
    gap       = mean_pred - frac_pos

    if abs(gap) < 0.05:
        summary = "well-calibrated"
    elif gap > 0:
        summary = "overconfident (predicted prob > observed rate)"
    else:
        summary = "underconfident (predicted prob < observed rate)"

    return {
        "mean_predicted_prob": round(mean_pred, 4),
        "fraction_positive":   round(frac_pos, 4),
        "overall_gap":         round(gap, 4),
        "calibration_summary": summary,
        "bin_reliability":     bin_data,
    }


# ============================================================================
# CALIBRATION CURVE + BRIER SCORE
# ============================================================================

def compute_and_plot_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str,
    n_bins: int = 10,
) -> Dict:
    """
    Compute Brier score and plot a reliability diagram with a probability histogram.

    The reliability diagram (calibration curve) compares the mean predicted
    probability in each bin against the observed fraction of positives.  A
    perfectly calibrated model falls on the diagonal.  The histogram beneath
    shows how predicted probabilities are distributed across the test set.

    Parameters
    ----------
    y_true      : ground-truth binary labels
    y_prob      : predicted probabilities for the positive class
    output_path : file path for the saved PNG figure
    n_bins      : number of probability bins (default 10)

    Returns
    -------
    dict with:
        brier_score : float — lower is better (0 = perfect, 0.25 = random)
        n_bins      : int
    """
    brier = float(brier_score_loss(y_true, y_prob))

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 9),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # ── Reliability diagram ──────────────────────────────────────────────────
    ax1.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")
    ax1.plot(
        mean_predicted_value, fraction_of_positives,
        "s-", color="#2563EB", lw=2, markersize=7,
        label=f"Model  (Brier = {brier:.4f})",
    )
    ax1.set_xlabel("Mean predicted probability", fontsize=11)
    ax1.set_ylabel("Fraction of positives", fontsize=11)
    ax1.set_title("Calibration Curve (Reliability Diagram)", fontsize=13,
                  fontweight="bold")
    ax1.legend(loc="lower right", fontsize=10)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.grid(alpha=0.3)

    # ── Predicted-probability histogram ─────────────────────────────────────
    ax2.hist(y_prob, bins=n_bins, range=(0, 1), color="#2563EB", alpha=0.7,
             edgecolor="white")
    ax2.set_xlabel("Predicted probability", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_xlim([0.0, 1.0])
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Calibration curve saved: {output_path}")

    return {
        "brier_score": round(brier, 4),
        "n_bins": n_bins,
    }


# ============================================================================
# PER-SAMPLE PREDICTIONS CSV
# ============================================================================

def save_predictions_csv(
    probs: np.ndarray,
    ai_preds: np.ndarray,
    biopsy_labels: np.ndarray,
    metadata_list: List[Dict],
    output_path: str,
) -> None:
    """Save per-sample probabilities, predictions, and ground-truth labels."""
    has_image_id = bool(metadata_list and "image_id" in metadata_list[0])
    fieldnames = ["sample_idx"]
    if has_image_id:
        fieldnames.append("image_id")
    fieldnames += ["biopsy_label", "ai_prob", "ai_pred", "correct"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, (prob, pred, label, meta) in enumerate(
            zip(probs, ai_preds, biopsy_labels, metadata_list)
        ):
            row: Dict = {
                "sample_idx":   i,
                "biopsy_label": int(label),
                "ai_prob":      round(float(prob), 4),
                "ai_pred":      int(pred),
                "correct":      int(pred) == int(label),
            }
            if has_image_id:
                row["image_id"] = meta.get("image_id", "")
            writer.writerow(row)

    print(f"  Predictions CSV saved: {output_path}")


# ============================================================================
# MAIN EVALUATION FUNCTION
# ============================================================================

def evaluate(
    model: nn.Module,
    test_dataloader: DataLoader,
    config: Optional[EvaluationConfig] = None,
) -> Dict[str, Dict]:
    """
    Full evaluation of a trained clinical AI model.

    The dataloader must return batches of (images, biopsy_labels, metadata).
    Clinician comparisons are computed automatically when the metadata dicts
    contain a ``clinician_diagnosis`` key with a non-None integer value.

    Parameters
    ----------
    model            : trained PyTorch model in eval-ready state
    test_dataloader  : DataLoader over the clinical test set
    config           : EvaluationConfig; uses defaults if None

    Returns
    -------
    report : dict   {"AI vs Biopsy": {...}, "Clinician vs Biopsy": {...}, ...}
    """
    if config is None:
        config = EvaluationConfig()

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("CLINICAL AI EVALUATION")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Run inference
    # ------------------------------------------------------------------
    print("\n[1/7] Running inference...")
    probs, ai_preds, biopsy_labels, metadata_list = run_inference(
        model, test_dataloader, config
    )
    print(f"  Samples evaluated: {len(biopsy_labels)}")
    print(f"  Positive (biopsy): {biopsy_labels.sum()} / {len(biopsy_labels)}")

    # ------------------------------------------------------------------
    # 2. Extract clinician labels (optional)
    # ------------------------------------------------------------------
    clinician_labels: Optional[np.ndarray] = None
    raw_cl = [
        m.get("clinician_diagnosis") if isinstance(m, dict) else None
        for m in metadata_list
    ]
    if all(v is not None for v in raw_cl):
        clinician_labels = np.array([int(v) for v in raw_cl])
        print(f"  Clinician labels found for all {len(clinician_labels)} samples")
    else:
        n_missing = sum(1 for v in raw_cl if v is None)
        print(f"  Clinician labels unavailable for {n_missing} samples — "
              "clinician comparisons will be skipped")

    # ------------------------------------------------------------------
    # 3. Compute metrics
    # ------------------------------------------------------------------
    print("\n[2/8] Computing metrics...")

    ai_vs_biopsy = compute_binary_metrics(
        biopsy_labels, ai_preds, y_prob=probs, label="AI vs Biopsy"
    )

    clinician_vs_biopsy: Optional[Dict] = None
    ai_vs_clinician: Optional[Dict] = None

    if clinician_labels is not None:
        clinician_vs_biopsy = compute_binary_metrics(
            biopsy_labels, clinician_labels, label="Clinician vs Biopsy"
        )
        ai_vs_clinician = compute_binary_metrics(
            clinician_labels, ai_preds, label="AI vs Clinician"
        )

    # ------------------------------------------------------------------
    # 4. Threshold analysis
    # ------------------------------------------------------------------
    threshold_results: Optional[Dict] = None
    calibration_results: Optional[Dict] = None
    if len(np.unique(biopsy_labels)) > 1:
        print("\n[3/8] Threshold analysis...")
        threshold_results = compute_threshold_analysis(biopsy_labels, probs)
        print(f"  Youden-optimal threshold : {threshold_results['best_threshold']}")
        print(f"  Youden's J               : {threshold_results['best_youden_j']}")
        print(f"  At optimal — Sensitivity : {threshold_results['best_sensitivity']}, "
              f"Specificity : {threshold_results['best_specificity']}")
        hdr = f"\n  {'Thresh':>7}  {'Sensitivity':>11}  {'Specificity':>11}  {'Precision':>9}  {'F1':>6}"
        print(hdr)
        print(f"  {'-'*52}")
        for row in threshold_results["threshold_table"]:
            tag = ""
            if abs(row["threshold"] - config.threshold) < 1e-6:
                tag = " <-- current"
            elif abs(row["threshold"] - threshold_results["best_threshold"]) < 1e-6:
                tag = " <-- Youden"
            print(f"  {row['threshold']:>7.3f}  {row['sensitivity']:>11.4f}  "
                  f"{row['specificity']:>11.4f}  {row['precision']:>9.4f}  "
                  f"{row['f1']:>6.4f}{tag}")

        # ------------------------------------------------------------------
        # 5. Calibration insight
        # ------------------------------------------------------------------
        print("\n[4/8] Calibration insight...")
        calibration_results = compute_calibration_insight(biopsy_labels, probs)
        print(f"  {calibration_results['calibration_summary']}")
        print(f"  Mean predicted prob = {calibration_results['mean_predicted_prob']:.4f}  |  "
              f"Fraction positive = {calibration_results['fraction_positive']:.4f}  |  "
              f"Gap = {calibration_results['overall_gap']:+.4f}")
        print(f"\n  {'Bin':>9}  {'N':>5}  {'Mean prob':>9}  {'Obs. rate':>9}  {'Gap':>7}")
        print(f"  {'-'*47}")
        for b in calibration_results["bin_reliability"]:
            print(f"  {b['bin']:>9}  {b['n']:>5}  {b['mean_prob']:>9.3f}  "
                  f"{b['observed_rate']:>9.3f}  {b['gap']:>+7.3f}")
    else:
        print("\n[3/8] Threshold analysis skipped (only one class present in labels).")
        print("[4/8] Calibration insight skipped.")

    # ------------------------------------------------------------------
    # 6. Confusion matrices
    # ------------------------------------------------------------------
    print("\n[5/8] Generating confusion matrices...")
    cm_path = os.path.join(config.output_dir, config.confusion_matrix_filename)
    confusion_matrices = compute_and_plot_confusion_matrices(
        ai_vs_biopsy=(biopsy_labels, ai_preds),
        clinician_vs_biopsy=(
            (biopsy_labels, clinician_labels)
            if clinician_labels is not None else None
        ),
        ai_vs_clinician=(
            (clinician_labels, ai_preds)
            if clinician_labels is not None else None
        ),
        output_path=cm_path,
    )

    # Print text confusion matrix for quick inspection
    cm_ab = confusion_matrices["AI vs Biopsy"]
    tn, fp, fn, tp = int(cm_ab[0, 0]), int(cm_ab[0, 1]), int(cm_ab[1, 0]), int(cm_ab[1, 1])
    print(f"\n  Confusion matrix — AI vs Biopsy:")
    print(f"    {'':<18}  Pred Neg  Pred Pos")
    print(f"    {'True Neg (benign)':<18}  {tn:8d}  {fp:8d}")
    print(f"    {'True Pos (malig.)':<18}  {fn:8d}  {tp:8d}")

    # ------------------------------------------------------------------
    # 7. ROC curve
    # ------------------------------------------------------------------
    print("\n[6/8] Plotting ROC curve...")
    roc_path = os.path.join(config.output_dir, config.roc_curve_filename)
    plot_roc_curve(
        y_true_biopsy=biopsy_labels,
        y_prob_ai=probs,
        y_pred_clinician=clinician_labels,
        output_path=roc_path,
    )

    # ------------------------------------------------------------------
    # 7. Calibration curve + Brier score
    # ------------------------------------------------------------------
    calibration_plot_results: Optional[Dict] = None
    cal_path = os.path.join(config.output_dir, config.calibration_filename)
    if len(np.unique(biopsy_labels)) > 1:
        print("\n[7/8] Computing Brier score and calibration curve...")
        calibration_plot_results = compute_and_plot_calibration(
            biopsy_labels, probs, output_path=cal_path
        )
        ai_vs_biopsy["brier_score"] = calibration_plot_results["brier_score"]
        print(f"  Brier score : {calibration_plot_results['brier_score']:.4f}  "
              "(0 = perfect, 0.25 = random)")
    else:
        print("\n[7/8] Calibration curve skipped (only one class present).")

    # ------------------------------------------------------------------
    # 8. Build & save report
    # ------------------------------------------------------------------
    print("\n[8/8] Saving reports...")
    report = build_metrics_report(ai_vs_biopsy, clinician_vs_biopsy, ai_vs_clinician)

    csv_path = os.path.join(config.output_dir, config.report_csv_filename)
    json_path = os.path.join(config.output_dir, "metrics_report.json")
    preds_csv_path = os.path.join(config.output_dir, config.predictions_csv_filename)
    save_report_csv(report, csv_path)
    save_report_json(report, json_path)
    save_predictions_csv(probs, ai_preds, biopsy_labels, metadata_list, preds_csv_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    _r = ai_vs_biopsy
    print(f"\n  Clinical metrics — AI vs Biopsy (threshold = {config.threshold})")
    print(f"  {'-'*40}")
    print(f"  Accuracy     : {_r['accuracy']}")
    print(f"  AUC-ROC      : {_r['roc_auc']}")
    print(f"  Sensitivity  : {_r['sensitivity']}  (recall for malignant)")
    print(f"  Specificity  : {_r['specificity']}")
    print(f"  Precision    : {_r['precision']}")
    print(f"  F1-score     : {_r['f1']}")
    print(f"  Kappa        : {_r['kappa']}")
    if threshold_results is not None:
        print(f"\n  Youden-optimal threshold : {threshold_results['best_threshold']}"
              f"  (J = {threshold_results['best_youden_j']},"
              f" Sens = {threshold_results['best_sensitivity']},"
              f" Spec = {threshold_results['best_specificity']})")
    if calibration_results is not None:
        print(f"  Calibration              : {calibration_results['calibration_summary']}")
    if calibration_plot_results is not None:
        print(f"  Brier Score              : {calibration_plot_results['brier_score']:.4f}")
    print(f"\n  Output directory : {os.path.abspath(config.output_dir)}")
    print(f"  Predictions CSV  : {preds_csv_path}")
    print(f"  Metrics CSV      : {csv_path}")
    print(f"  Metrics JSON     : {json_path}")
    print(f"  ROC curve        : {roc_path}")
    print(f"  Confusion matrix : {cm_path}")
    if calibration_plot_results is not None:
        print(f"  Calibration plot : {cal_path}")

    return report
