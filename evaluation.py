"""
src/evaluation.py
=================
Metrics computation and visualisation for all trained models:
  - accuracy, precision, recall, F1-score, ROC-AUC
  - confusion matrix
  - ROC curves
  - comparative bar charts
  - learning curves
"""

from __future__ import annotations

import io
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import label_binarize

# ── Style ──────────────────────────────────────────────────────────────────────
PALETTE = [
    "#7C3AED", "#3B82F6", "#10B981", "#F59E0B",
    "#EF4444", "#8B5CF6", "#06B6D4",
]
_DARK_BG   = "#0F0F1A"
_PANEL_BG  = "#1A1A2E"
_TEXT      = "#E2E8F0"
_GRID      = "#2D2D44"


def _set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor":  _DARK_BG,
        "axes.facecolor":    _PANEL_BG,
        "axes.edgecolor":    _GRID,
        "axes.labelcolor":   _TEXT,
        "xtick.color":       _TEXT,
        "ytick.color":       _TEXT,
        "text.color":        _TEXT,
        "grid.color":        _GRID,
        "legend.facecolor":  _PANEL_BG,
        "legend.edgecolor":  _GRID,
        "font.family":       "DejaVu Sans",
    })


# ── Core metrics ───────────────────────────────────────────────────────────────

def compute_metrics(
    name: str,
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    train_time: float = 0.0,
    n_classes: int = 2,
) -> dict:
    """
    Compute all key metrics for one fitted model.

    Returns
    -------
    dict with accuracy, precision, recall, f1, roc_auc, train_time.
    """
    y_pred = model.predict(X_test)
    y_prob = _get_proba(model, X_test, n_classes)

    avg = "binary" if n_classes == 2 else "macro"

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
    rec  = recall_score(y_test, y_pred, average=avg, zero_division=0)
    f1   = f1_score(y_test, y_pred, average=avg, zero_division=0)
    auc  = _compute_auc(y_test, y_prob, n_classes)

    return {
        "Model":      name,
        "Accuracy":   round(acc,  4),
        "Precision":  round(prec, 4),
        "Recall":     round(rec,  4),
        "F1-Score":   round(f1,   4),
        "ROC-AUC":    round(auc,  4),
        "Train Time (s)": round(train_time, 2),
    }


def compute_all_metrics(
    trained_models: Dict[str, tuple],
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int = 2,
) -> pd.DataFrame:
    """
    Run compute_metrics for every trained model and return a DataFrame.

    Parameters
    ----------
    trained_models : dict[name → (fitted_model, train_time)]
    """
    rows = []
    for name, (model, train_time) in trained_models.items():
        row = compute_metrics(name, model, X_test, y_test, train_time, n_classes)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Model")
    return df.sort_values("F1-Score", ascending=False)


def get_classification_report(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> str:
    """Return sklearn classification_report as string."""
    y_pred = model.predict(X_test)
    return classification_report(y_test, y_pred, target_names=class_names,
                                  zero_division=0)


# ── Visualisations ─────────────────────────────────────────────────────────────

def plot_metrics_comparison(metrics_df: pd.DataFrame) -> plt.Figure:
    """Bar chart comparing all models across all metrics."""
    _set_style()
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    df = metrics_df[metric_cols].reset_index()
    df_melt = df.melt(id_vars="Model", var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(14, 6), facecolor=_DARK_BG)
    ax.set_facecolor(_PANEL_BG)

    models   = df["Model"].tolist()
    metrics  = metric_cols
    x        = np.arange(len(models))
    width    = 0.15

    for i, (metric, color) in enumerate(zip(metrics, PALETTE)):
        vals = df[metric].values
        bars = ax.bar(
            x + i * width - width * 2,
            vals, width,
            label=metric,
            color=color,
            alpha=0.88,
        )
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.004,
                f"{h:.3f}",
                ha="center", va="bottom",
                fontsize=6.5, color=_TEXT,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9, color=_TEXT)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold",
                 color=_TEXT, pad=12)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.4)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color=_GRID)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None,
    model_name: str = "",
) -> plt.Figure:
    """Styled confusion matrix heatmap."""
    _set_style()
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    n  = cm.shape[0]
    names = class_names or [str(i) for i in range(n)]

    fig, ax = plt.subplots(figsize=(max(5, n * 1.4), max(4, n * 1.2)),
                            facecolor=_DARK_BG)
    ax.set_facecolor(_PANEL_BG)

    sns.heatmap(
        cm, annot=True, fmt="d",
        xticklabels=names, yticklabels=names,
        cmap="Purples",
        linewidths=0.5, linecolor=_GRID,
        ax=ax,
        annot_kws={"size": 11, "weight": "bold"},
    )
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=12,
                 fontweight="bold", color=_TEXT, pad=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(_TEXT)

    fig.tight_layout()
    return fig


def plot_roc_curves(
    trained_models: Dict[str, tuple],
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int = 2,
) -> plt.Figure:
    """ROC curves for all models (binary classification only)."""
    _set_style()
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=_DARK_BG)
    ax.set_facecolor(_PANEL_BG)

    for (name, (model, _)), color in zip(trained_models.items(), PALETTE):
        y_prob = _get_proba(model, X_test, n_classes)
        if y_prob is None:
            continue
        if n_classes == 2:
            prob = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
            fpr, tpr, _ = roc_curve(y_test, prob)
            auc = roc_auc_score(y_test, prob)
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("ROC Curves", fontsize=13, fontweight="bold",
                 color=_TEXT, pad=12)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.4)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, color=_GRID)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3, color=_GRID)

    fig.tight_layout()
    return fig


def plot_learning_curve(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = "",
    cv: int = 5,
    n_jobs: int = -1,
) -> plt.Figure:
    """Bias-variance learning curve plot."""
    _set_style()
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        cv=cv, scoring="f1_macro",
        n_jobs=n_jobs,
        train_sizes=np.linspace(0.1, 1.0, 8),
    )

    ts_mean = train_scores.mean(axis=1)
    ts_std  = train_scores.std(axis=1)
    vs_mean = val_scores.mean(axis=1)
    vs_std  = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=_DARK_BG)
    ax.set_facecolor(_PANEL_BG)

    ax.plot(train_sizes, ts_mean, "o-", color=PALETTE[0], lw=2, label="Training score")
    ax.fill_between(train_sizes, ts_mean - ts_std, ts_mean + ts_std,
                     alpha=0.12, color=PALETTE[0])
    ax.plot(train_sizes, vs_mean, "o-", color=PALETTE[1], lw=2, label="CV score")
    ax.fill_between(train_sizes, vs_mean - vs_std, vs_mean + vs_std,
                     alpha=0.12, color=PALETTE[1])

    ax.set_xlabel("Training examples", fontsize=10)
    ax.set_ylabel("F1 (macro)", fontsize=10)
    ax.set_title(f"Learning Curve – {model_name}", fontsize=12,
                 fontweight="bold", color=_TEXT, pad=10)
    ax.legend(fontsize=9, framealpha=0.4)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, color=_GRID)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    return fig


def plot_feature_importance(
    model,
    feature_names: List[str],
    top_n: int = 20,
    model_name: str = "",
) -> Optional[plt.Figure]:
    """Horizontal bar chart of feature importances (tree-based models only)."""
    _set_style()
    if not hasattr(model, "feature_importances_"):
        return None

    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    names   = [feature_names[i] for i in indices]
    vals    = importances[indices]

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.35)), facecolor=_DARK_BG)
    ax.set_facecolor(_PANEL_BG)

    colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(vals)))
    ax.barh(names, vals, color=colors, edgecolor="none")
    ax.set_xlabel("Importance", fontsize=10)
    ax.set_title(f"Feature Importances – {model_name}", fontsize=12,
                 fontweight="bold", color=_TEXT, pad=10)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3, color=_GRID)
    ax.set_axisbelow(True)

    fig.tight_layout()
    return fig


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_proba(model, X_test, n_classes: int):
    """Return probability array or None."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        if n_classes == 2:
            # Convert to [neg, pos] proba-like shape
            prob = 1 / (1 + np.exp(-scores))
            return np.column_stack([1 - prob, prob])
    return None


def _compute_auc(y_test, y_prob, n_classes: int) -> float:
    if y_prob is None:
        return 0.0
    try:
        if n_classes == 2:
            prob = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
            return roc_auc_score(y_test, prob)
        # Multiclass OvR
        classes = sorted(np.unique(y_test))
        y_bin = label_binarize(y_test, classes=classes)
        return roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
    except Exception:
        return 0.0
