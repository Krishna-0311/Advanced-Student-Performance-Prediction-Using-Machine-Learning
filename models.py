"""
src/models.py
=============
Defines, trains, and returns all ML models for student performance prediction.

Models
------
Baseline  : Logistic Regression, Decision Tree, SVM (RBF)
Ensemble  : Random Forest, Gradient Boosting, XGBoost
Advanced  : Voting Classifier (RF + GBM + XGB)

Tuning    : RandomizedSearchCV on top-3 ensemble models
"""

from __future__ import annotations

import logging
from time import perf_counter
from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

# ── Default hyper-parameter search spaces ─────────────────────────────────────

RF_PARAM_GRID = {
    "n_estimators":     [100, 200, 300],
    "max_depth":        [None, 6, 10, 15],
    "min_samples_split":[2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features":     ["sqrt", "log2"],
}

GBM_PARAM_GRID = {
    "n_estimators":  [100, 200, 300],
    "learning_rate": [0.05, 0.1, 0.15],
    "max_depth":     [3, 4, 5, 6],
    "subsample":     [0.7, 0.85, 1.0],
}

XGB_PARAM_GRID = {
    "n_estimators":   [100, 200, 300],
    "learning_rate":  [0.03, 0.05, 0.1, 0.15],
    "max_depth":      [3, 4, 5, 6],
    "subsample":      [0.7, 0.85, 1.0],
    "colsample_bytree": [0.7, 0.85, 1.0],
    "reg_alpha":      [0, 0.1, 0.5],
    "reg_lambda":     [1, 1.5, 2],
}


# ── Model registry ─────────────────────────────────────────────────────────────

def get_all_models(
    n_classes: int = 2,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """
    Return a dict of unfitted estimators.

    Parameters
    ----------
    n_classes : int
        2 for binary (Pass/Fail), 5 for multiclass (A/B/C/D/F).
    random_state : int
        Global seed for reproducibility.
    n_jobs : int
        Parallelism for ensemble models.

    Returns
    -------
    dict[str, estimator]
    """
    is_multiclass = n_classes > 2

    xgb_objective = "multi:softprob" if is_multiclass else "binary:logistic"
    xgb_eval = "mlogloss" if is_multiclass else "logloss"

    models: Dict[str, Any] = {
        # ── Baseline ──────────────────────────────────────
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=random_state,
            n_jobs=n_jobs,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=5,
            random_state=random_state,
        ),
        "SVM (RBF)": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=random_state,
        ),
        # ── Ensemble ──────────────────────────────────────
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=3,
            random_state=random_state,
            n_jobs=n_jobs,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.85,
            random_state=random_state,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            use_label_encoder=False,
            eval_metric=xgb_eval,
            objective=xgb_objective,
            num_class=n_classes if is_multiclass else None,
            random_state=random_state,
            n_jobs=n_jobs,
            verbosity=0,
        ),
        # ── Advanced Ensemble ─────────────────────────────
        "Voting Classifier": _build_voting_classifier(
            n_classes, random_state, n_jobs
        ),
    }

    return models


def _build_voting_classifier(
    n_classes: int,
    random_state: int,
    n_jobs: int,
) -> VotingClassifier:
    """Build a soft-voting ensemble from RF, GBM and XGB."""
    xgb_eval = "mlogloss" if n_classes > 2 else "logloss"
    xgb_obj  = "multi:softprob" if n_classes > 2 else "binary:logistic"

    rf  = RandomForestClassifier(n_estimators=150, max_depth=10,
                                  random_state=random_state, n_jobs=n_jobs)
    gbm = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                      max_depth=4, random_state=random_state)
    xgb = XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=5,
                         use_label_encoder=False, eval_metric=xgb_eval,
                         objective=xgb_obj,
                         num_class=n_classes if n_classes > 2 else None,
                         random_state=random_state, n_jobs=n_jobs, verbosity=0)

    return VotingClassifier(
        estimators=[("rf", rf), ("gbm", gbm), ("xgb", xgb)],
        voting="soft",
        n_jobs=n_jobs,
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str = "",
) -> tuple[Any, float]:
    """Fit a single model and return (fitted_model, train_time_seconds)."""
    t0 = perf_counter()
    model.fit(X_train, y_train)
    elapsed = perf_counter() - t0
    logger.info("Trained %-25s in %.2f s", model_name, elapsed)
    return model, elapsed


def train_all_models(
    models: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Dict[str, tuple[Any, float]]:
    """
    Train all models sequentially.

    Returns
    -------
    dict[name → (fitted_model, train_time)]
    """
    results: Dict[str, tuple[Any, float]] = {}
    for name, model in models.items():
        fitted, elapsed = train_model(model, X_train, y_train, name)
        results[name] = (fitted, elapsed)
    return results


def tune_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int = 2,
    n_iter: int = 20,
    cv: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Optional[Any]:
    """
    Perform RandomizedSearchCV for the given model.

    Parameters
    ----------
    model_name : str
        One of "Random Forest", "Gradient Boosting", "XGBoost".

    Returns
    -------
    Best estimator or None if model_name not supported.
    """
    if model_name == "Random Forest":
        base = RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)
        grid = RF_PARAM_GRID
    elif model_name == "Gradient Boosting":
        base = GradientBoostingClassifier(random_state=random_state)
        grid = GBM_PARAM_GRID
    elif model_name == "XGBoost":
        xgb_eval = "mlogloss" if n_classes > 2 else "logloss"
        xgb_obj  = "multi:softprob" if n_classes > 2 else "binary:logistic"
        base = XGBClassifier(
            use_label_encoder=False,
            eval_metric=xgb_eval,
            objective=xgb_obj,
            num_class=n_classes if n_classes > 2 else None,
            random_state=random_state, n_jobs=n_jobs, verbosity=0,
        )
        grid = XGB_PARAM_GRID
    else:
        logger.warning("Tuning not supported for '%s'", model_name)
        return None

    scorer = "roc_auc" if n_classes == 2 else "f1_macro"
    search = RandomizedSearchCV(
        base, grid,
        n_iter=n_iter,
        cv=cv,
        scoring=scorer,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=0,
    )
    t0 = perf_counter()
    search.fit(X_train, y_train)
    logger.info(
        "Tuned %s – best %s=%.4f  (%.1f s)",
        model_name, scorer, search.best_score_, perf_counter() - t0
    )
    return search.best_estimator_
