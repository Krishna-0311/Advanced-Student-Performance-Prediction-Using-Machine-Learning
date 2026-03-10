"""
src/explainability.py
=====================
SHAP-based explainability for tree-based and linear ML models.

Provides:
 - Global summary plots (beeswarm, bar)
 - Waterfall plot (single prediction)
 - Force plot (HTML)
 - Dependence plots
 - SHAP values as DataFrame
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)

_DARK_BG  = "#0F0F1A"
_PANEL_BG = "#1A1A2E"
_TEXT     = "#E2E8F0"


def _set_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": _DARK_BG,
        "axes.facecolor":   _PANEL_BG,
        "text.color":       _TEXT,
        "axes.labelcolor":  _TEXT,
        "xtick.color":      _TEXT,
        "ytick.color":      _TEXT,
    })


def _get_explainer(
    model,
    X_background: np.ndarray,
    model_name: str,
) -> shap.Explainer:
    """
    Choose the most appropriate SHAP explainer for the model type.
    Falls back to KernelExplainer if tree/linear not applicable.
    """
    try:
        # Tree-based
        for cls_name in ["RandomForest", "GradientBoosting",
                          "XGB", "DecisionTree"]:
            if cls_name.lower() in type(model).__name__.lower():
                return shap.TreeExplainer(model)

        # Linear
        if "Logistic" in type(model).__name__ or "LinearSVC" in type(model).__name__:
            return shap.LinearExplainer(model, X_background)

    except Exception as e:
        logger.warning("Specialised explainer failed (%s) – falling back to Kernel", e)

    # Universal fallback (slow – subsample background)
    bg = shap.sample(X_background, min(100, len(X_background)))
    return shap.KernelExplainer(model.predict_proba, bg)


def compute_shap_values(
    model,
    X: pd.DataFrame | np.ndarray,
    X_background: pd.DataFrame | np.ndarray | None = None,
    model_name: str = "",
    max_samples: int = 300,
) -> tuple[shap.Explainer, np.ndarray]:
    """
    Compute SHAP values for the given model and data.

    Parameters
    ----------
    model : fitted estimator
    X : feature matrix (to explain)
    X_background : background dataset for KernelExplainer
    model_name : str for logging
    max_samples : subsample X to limit compute time

    Returns
    -------
    explainer : shap.Explainer
    shap_vals : np.ndarray  [n_samples × n_features] or [n_classes × n_samples × n_features]
    """
    if X_background is None:
        X_background = X

    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    bg    = X_background.values if isinstance(X_background, pd.DataFrame) else X_background

    # Subsample
    n = min(max_samples, len(X_arr))
    idx = np.random.default_rng(42).choice(len(X_arr), n, replace=False)
    X_sample = X_arr[idx]

    explainer = _get_explainer(model, bg, model_name)

    try:
        sv = explainer.shap_values(X_sample)
    except Exception as e:
        logger.error("SHAP computation failed: %s", e)
        raise

    return explainer, sv


def shap_summary_plot(
    shap_values: np.ndarray,
    X: pd.DataFrame | np.ndarray,
    feature_names: List[str],
    plot_type: str = "bar",
    class_idx: int = 1,
    max_display: int = 20,
) -> plt.Figure:
    """
    Global SHAP summary plot.

    Parameters
    ----------
    plot_type : "bar" | "beeswarm"
    class_idx : for multiclass, which class to visualise (default=1)
    """
    _set_style()
    X_arr = X.values if isinstance(X, pd.DataFrame) else X

    # Handle multi-output SHAP (list of arrays from tree classifiers)
    sv = shap_values
    if isinstance(sv, list):
        sv = sv[min(class_idx, len(sv) - 1)]

    fig, ax = plt.subplots(figsize=(10, 6), facecolor=_DARK_BG)
    plt.sca(ax)

    shap.summary_plot(
        sv, X_arr,
        feature_names=feature_names,
        plot_type=plot_type,
        max_display=max_display,
        show=False,
        color_bar=True,
    )

    plt.gcf().set_facecolor(_DARK_BG)
    for ax_i in plt.gcf().get_axes():
        ax_i.set_facecolor(_PANEL_BG)
        ax_i.tick_params(colors=_TEXT)
        ax_i.yaxis.label.set_color(_TEXT)
        ax_i.xaxis.label.set_color(_TEXT)
        ax_i.title.set_color(_TEXT)

    plt.tight_layout()
    return plt.gcf()


def shap_waterfall_plot(
    explainer: shap.Explainer,
    X_instance: np.ndarray,
    feature_names: List[str],
    class_idx: int = 1,
) -> plt.Figure:
    """
    Local waterfall plot explaining one prediction.

    Parameters
    ----------
    X_instance : shape (n_features,) – single sample
    """
    _set_style()
    try:
        ex = explainer(X_instance.reshape(1, -1))
        if hasattr(ex, "values"):
            vals = ex.values
            # For multi-output explainers, select the right class
            if len(vals.shape) == 3:
                vals = vals[:, :, class_idx]
            # Build Explanation object for waterfall
            exp = shap.Explanation(
                values=vals[0],
                base_values=(ex.base_values[0] if ex.base_values.ndim > 1
                              else ex.base_values),
                data=X_instance,
                feature_names=feature_names,
            )
        else:
            sv = explainer.shap_values(X_instance.reshape(1, -1))
            if isinstance(sv, list):
                sv = sv[min(class_idx, len(sv) - 1)]
            base = (explainer.expected_value[min(class_idx, len(explainer.expected_value) - 1)]
                    if hasattr(explainer.expected_value, "__len__")
                    else explainer.expected_value)
            exp = shap.Explanation(
                values=sv[0],
                base_values=base,
                data=X_instance,
                feature_names=feature_names,
            )

        fig, ax = plt.subplots(figsize=(10, 6), facecolor=_DARK_BG)
        plt.sca(ax)
        shap.plots.waterfall(exp, max_display=15, show=False)
        fig = plt.gcf()
        fig.set_facecolor(_DARK_BG)
        for ax_i in fig.get_axes():
            ax_i.set_facecolor(_PANEL_BG)
        fig.tight_layout()
        return fig
    except Exception as e:
        logger.warning("Waterfall plot failed: %s", e)
        return _fallback_bar_plot(
            explainer.shap_values(X_instance.reshape(1, -1)),
            feature_names, class_idx
        )


def shap_dependence_plot(
    shap_values: np.ndarray,
    X: pd.DataFrame | np.ndarray,
    feature_names: List[str],
    feature: str,
    interaction_feature: str = "auto",
    class_idx: int = 1,
) -> plt.Figure:
    """SHAP dependence plot for a single feature."""
    _set_style()
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    sv    = shap_values
    if isinstance(sv, list):
        sv = sv[min(class_idx, len(sv) - 1)]

    feat_idx = feature_names.index(feature) if feature in feature_names else 0

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=_DARK_BG)
    plt.sca(ax)
    shap.dependence_plot(
        feat_idx, sv, X_arr,
        feature_names=feature_names,
        interaction_index=interaction_feature,
        show=False,
        ax=ax,
    )
    fig.set_facecolor(_DARK_BG)
    ax.set_facecolor(_PANEL_BG)
    ax.tick_params(colors=_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.xaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)
    fig.tight_layout()
    return fig


def shap_to_dataframe(shap_vals, feature_names, class_idx=0):
    import numpy as np
    import pandas as pd

    shap_vals = np.array(shap_vals)

    # If 3D (multi-class case)
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[class_idx]

    # If 2D (single sample but still 2D)
    if shap_vals.ndim == 2:
        shap_vals = shap_vals[0]

    # Now it must be 1D
    shap_vals = shap_vals.flatten()

    df = pd.DataFrame({
        "feature": feature_names,
        "shap_value": shap_vals
    })

    df["abs_shap"] = df["shap_value"].abs()
    df = df.sort_values("abs_shap", ascending=False)

    return df


def _fallback_bar_plot(sv, feature_names: List[str], class_idx: int) -> plt.Figure:
    """Simple bar fallback when waterfall fails."""
    _set_style()
    if isinstance(sv, list):
        sv = sv[min(class_idx, len(sv) - 1)]
    vals  = sv[0]
    names = feature_names[:len(vals)]
    idx   = np.argsort(np.abs(vals))[-15:]

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=_DARK_BG)
    ax.set_facecolor(_PANEL_BG)
    colors = ["#7C3AED" if v >= 0 else "#EF4444" for v in vals[idx]]
    ax.barh([names[i] for i in idx], vals[idx], color=colors)
    ax.set_xlabel("SHAP value", color=_TEXT)
    ax.set_title("Local Feature Contributions", color=_TEXT, fontsize=12)
    fig.tight_layout()
    return fig
