"""Metrics, threshold tuning, plots and the experiment log.

Average precision (PR-AUC) is the headline metric. With ~5% positives, accuracy
and ROC-AUC both look flattering while the model catches almost nothing.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import config

METRIC_COLUMNS = [
    "average_precision",
    "roc_auc",
    "f1",
    "precision",
    "recall",
    "accuracy",
    "threshold",
]
EXPERIMENT_COLUMNS = ["timestamp", "model", "split"] + METRIC_COLUMNS + ["notes"]


# --- metrics ---------------------------------------------------------------


def evaluate_predictions(y_true, y_proba, threshold: float = 0.5) -> dict:
    """Metrics for one split.

    average_precision and roc_auc are threshold-free; f1/precision/recall/accuracy
    are computed at `threshold`. Call this once per split - don't wrap train and
    val into a single call.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "average_precision": average_precision_score(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "threshold": float(threshold),
    }


def tune_threshold(y_true, y_proba, beta: float = 1.0) -> Tuple[float, float]:
    """Threshold that maximises F-beta on this split. Returns (threshold, score).

    Tune on validation, never on test. beta=1 is plain F1; beta=2 weights recall
    higher, which is what you want if missing a scam costs more than a false alarm.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # precision/recall have one more element than thresholds; drop the last point.
    precision, recall = precision[:-1], recall[:-1]
    denominator = (beta**2 * precision) + recall
    scores = np.where(
        denominator > 0,
        (1 + beta**2) * precision * recall / np.where(denominator == 0, 1, denominator),
        0.0,
    )
    best = int(np.argmax(scores))
    return float(thresholds[best]), float(scores[best])


def threshold_for_precision(y_true, y_proba, min_precision: float = 0.9) -> Tuple[float, float]:
    """Lowest threshold that still reaches `min_precision`. Returns (threshold, recall).

    The screening question: "if we only auto-flag postings we're 90% sure about,
    how many scams do we still catch?" Returns (1.0, 0.0) if unreachable.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    precision, recall = precision[:-1], recall[:-1]
    feasible = np.flatnonzero(precision >= min_precision)
    if feasible.size == 0:
        return 1.0, 0.0
    best = feasible[int(np.argmax(recall[feasible]))]
    return float(thresholds[best]), float(recall[best])


# --- plots -----------------------------------------------------------------


def _save(fig, save_as: Optional[str]) -> None:
    if save_as is None:
        return
    path = Path(save_as)
    if not path.is_absolute():
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        path = config.FIGURES_DIR / save_as
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")


def plot_confusion_matrix(
    y_true, y_pred, title: str = "Confusion matrix", save_as: Optional[str] = None
):
    """Counts, with the share of the true class annotated underneath."""
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(matrix, cmap="Blues")

    row_totals = matrix.sum(axis=1, keepdims=True)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            share = matrix[i, j] / row_totals[i, 0] if row_totals[i, 0] else 0
            ax.text(
                j,
                i,
                f"{matrix[i, j]:,}\n{share:.1%}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > matrix.max() / 2 else "black",
            )

    ax.set_xticks([0, 1], ["pred real", "pred fake"])
    ax.set_yticks([0, 1], ["true real", "true fake"])
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, save_as)
    return ax


def plot_pr_curve(
    y_true, y_proba, label: Optional[str] = None, ax=None, save_as: Optional[str] = None
):
    """Precision-recall curve with AP in the legend.

    Pass the returned ax back in to overlay several models on one figure.
    """
    y_true = np.asarray(y_true)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    average_precision = average_precision_score(y_true, y_proba)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        baseline = y_true.mean()
        ax.axhline(
            baseline,
            linestyle="--",
            color="grey",
            linewidth=1,
            label=f"random ({baseline:.3f})",
        )
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall curve")
        ax.set_ylim(0, 1.02)

    ax.plot(recall, precision, label=f"{label or 'model'} (AP={average_precision:.3f})")
    ax.legend(loc="lower left", fontsize=8)
    _save(ax.figure, save_as)
    return ax


def plot_roc_curve(
    y_true, y_proba, label: Optional[str] = None, ax=None, save_as: Optional[str] = None
):
    """ROC curve with AUC in the legend. Reported for comparability, not for ranking."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="random")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC curve")

    ax.plot(fpr, tpr, label=f"{label or 'model'} (AUC={auc:.3f})")
    ax.legend(loc="lower right", fontsize=8)
    _save(ax.figure, save_as)
    return ax


# --- error analysis --------------------------------------------------------


def get_errors(
    df: pd.DataFrame,
    y_true,
    y_proba,
    threshold: float,
    kind: str = "false_negatives",
    top_n: int = 10,
) -> pd.DataFrame:
    """Rows the model got wrong, sorted by how confidently it was wrong.

    "false_negatives" are the scams that slipped through - read a few and ask why
    they look legitimate. "false_positives" are the real postings we would annoy.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    if kind == "false_negatives":
        mask = (y_true == 1) & (y_pred == 0)
        order = np.argsort(y_proba)  # most confidently wrong = lowest score
    elif kind == "false_positives":
        mask = (y_true == 0) & (y_pred == 1)
        order = np.argsort(-y_proba)
    else:
        raise ValueError(f"kind must be 'false_negatives' or 'false_positives', got {kind!r}")

    selected = [i for i in order if mask[i]][:top_n]
    errors = df.iloc[selected].copy()
    errors.insert(0, "predicted_proba", y_proba[selected])
    return errors


# --- experiment log --------------------------------------------------------


def log_experiment(
    name: str,
    metrics: dict,
    split: str = "val",
    notes: str = "",
    path: Path = config.EXPERIMENTS_CSV,
) -> dict:
    """Append one row to reports/experiments.csv, creating it with a header if needed."""
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": name,
        "split": split,
    }
    row.update({key: metrics.get(key) for key in METRIC_COLUMNS})
    row["notes"] = notes

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    pd.DataFrame([row], columns=EXPERIMENT_COLUMNS).to_csv(
        path, mode="a", header=write_header, index=False
    )
    return row


def load_experiments(path: Path = config.EXPERIMENTS_CSV) -> pd.DataFrame:
    """Read the experiment log back, best average precision first."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=EXPERIMENT_COLUMNS)
    df = pd.read_csv(path)
    if df.empty:
        return df
    return df.sort_values("average_precision", ascending=False).reset_index(drop=True)
