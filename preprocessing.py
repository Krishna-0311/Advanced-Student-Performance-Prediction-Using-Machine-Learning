"""
src/preprocessing.py
====================
Data loading, cleaning, encoding, scaling, SMOTE oversampling,
and train-test split utilities for the student performance dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)

# ── Columns ──────────────────────────────────────────────────────────────────
CATEGORICAL_COLS = [
    "gender", "address", "family_size", "parent_status",
    "mother_job", "father_job",
]

BINARY_COLS = [
    "extra_support", "family_support", "paid_classes", "activities",
    "nursery", "higher_ed_aspiration", "internet_access", "romantic",
]

ORDINAL_COLS = [
    "study_time", "travel_time", "free_time", "go_out",
    "dalc", "walc", "health",
    "mother_education", "father_education",
]

NUMERIC_COLS = [
    "age", "past_failures", "absences",
    "G1", "G2",
    "login_frequency", "assignment_submission_rate",
    "attendance_rate", "engagement_score",
    "forum_posts", "resource_access",
]

TARGET_BINARY        = "pass_fail"
TARGET_MULTICLASS    = "grade_label"
TARGET_REGRESSION    = "G3"
DROP_COLS            = ["G3", "grade_label", "pass_fail"]  # target leakage guard
GRADE_ORDER          = ["F", "D", "C", "B", "A"]


# ── Public API ────────────────────────────────────────────────────────────────

def load_raw_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load CSV or auto-generate if not found."""
    if csv_path is None:
        csv_path = Path(__file__).parent.parent / "data" / "students.csv"

    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.info("Dataset not found – generating synthetic data …")
        from data.generate_dataset import generate_dataset  # type: ignore
        df = generate_dataset(n_samples=1200, save=True)
    else:
        df = pd.read_csv(csv_path)

    logger.info("Loaded %d rows × %d cols", len(df), df.shape[1])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values and basic type corrections."""
    df = df.copy()

    # Numeric columns – fill NaN with median
    for col in NUMERIC_COLS + ORDINAL_COLS:
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    # Categorical columns – fill NaN with mode
    for col in CATEGORICAL_COLS + BINARY_COLS:
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)

    logger.info("Data cleaned – zero nulls: %s", df.isnull().sum().sum() == 0)
    return df


def encode_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Label-encode categorical columns.

    Returns
    -------
    df_encoded : pd.DataFrame
        Encoded dataframe.
    encoders : dict
        Fitted LabelEncoder per column for inverse-transform later.
    """
    df = df.copy()
    encoders: dict[str, LabelEncoder] = {}

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # grade_label ordinal encode
    if TARGET_MULTICLASS in df.columns:
        le_grade = LabelEncoder()
        le_grade.classes_ = np.array(GRADE_ORDER)
        df[TARGET_MULTICLASS] = pd.Categorical(
            df[TARGET_MULTICLASS], categories=GRADE_ORDER
        ).codes
        encoders[TARGET_MULTICLASS] = le_grade

    return df, encoders


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    method: Literal["standard", "minmax"] = "standard",
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler | MinMaxScaler]:
    """Fit scaler on train set, transform both sets."""
    Scaler = StandardScaler if method == "standard" else MinMaxScaler
    scaler = Scaler()

    X_train_s = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_s = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    return X_train_s, X_test_s, scaler


def prepare_data(
    df: pd.DataFrame,
    target: Literal["pass_fail", "grade_label"] = "pass_fail",
    test_size: float = 0.20,
    apply_smote: bool = True,
    scale_method: Literal["standard", "minmax"] = "standard",
    random_state: int = 42,
) -> dict:
    """
    Full preprocessing pipeline.

    Returns
    -------
    dict with keys:
        X_train, X_test, y_train, y_test,
        feature_names, encoders, scaler,
        X_train_raw, X_test_raw   (unscaled, for SHAP)
    """
    df = clean_data(df)
    df, encoders = encode_features(df)

    y = df[target].values.astype(int)

    # Drop all target columns before building X
    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols]

    X_train_r, X_test_r, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    X_train_s, X_test_s, scaler = scale_features(X_train_r, X_test_r, scale_method)

    if apply_smote:
        try:
            smote = SMOTE(random_state=random_state, k_neighbors=5)
            X_train_s_arr, y_train = smote.fit_resample(X_train_s.values, y_train)
            X_train_s = pd.DataFrame(X_train_s_arr, columns=X_train_s.columns)
        except ValueError as exc:
            logger.warning("SMOTE skipped: %s", exc)

    return {
        "X_train":      X_train_s,
        "X_test":       X_test_s,
        "y_train":      y_train,
        "y_test":       y_test,
        "feature_names": list(X.columns),
        "encoders":     encoders,
        "scaler":       scaler,
        "X_train_raw":  X_train_r.reset_index(drop=True),
        "X_test_raw":   X_test_r.reset_index(drop=True),
        "n_classes":    len(np.unique(y)),
        "target":       target,
    }
