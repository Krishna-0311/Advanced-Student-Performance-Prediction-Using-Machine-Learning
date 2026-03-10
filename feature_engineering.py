"""
src/feature_engineering.py
===========================
Derives composite and interaction features from the raw student dataset
to improve model signal, then appends them to the feature matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Feature groups ────────────────────────────────────────────────────────────

LMS_FEATURES = [
    "attendance_rate",
    "assignment_submission_rate",
    "engagement_score",
    "login_frequency",
    "forum_posts",
    "resource_access",
]

ACADEMIC_FEATURES = [
    "G1",
    "G2",
    "study_time",
    "past_failures",
    "absences",
]


def add_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-specific composite and interaction features.

    New columns created
    -------------------
    learning_index        : weighted composite of LMS engagement signals
    grade_momentum        : G2 - G1  (improvement/decline trend)
    academic_risk_score   : combines failures, absences, low grades
    parental_edu_avg      : average of mother + father education
    digital_engagement    : login_frequency × resource_access (normalised)
    study_efficiency      : study_time / (absences + 1)
    social_index          : go_out + free_time - dalc - walc (net social balance)
    behavioural_composite : attendance × submission_rate × engagement_score
    G1_sq, G2_sq          : polynomial (quadratic) terms for period grades
    """
    df = df.copy()

    # 1. Learning Index (0-10 scale)
    df["learning_index"] = (
        0.35 * _safe_col(df, "attendance_rate", 0) * 10
        + 0.30 * _safe_col(df, "assignment_submission_rate", 0) * 10
        + 0.20 * _safe_col(df, "engagement_score", 0)
        + 0.10 * _safe_col(df, "login_frequency", 0) / 3.0
        + 0.05 * _safe_col(df, "forum_posts", 0) / 2.5
    ).clip(0, 10)

    # 2. Grade momentum (G2 - G1)
    if "G1" in df.columns and "G2" in df.columns:
        df["grade_momentum"] = df["G2"] - df["G1"]

    # 3. Academic risk score (higher = more at risk)
    df["academic_risk_score"] = (
        _safe_col(df, "past_failures", 0) * 3
        + (_safe_col(df, "absences", 0) / 10.0)
        + (1 - _safe_col(df, "attendance_rate", 0.8)) * 5
        + (1 - _safe_col(df, "assignment_submission_rate", 0.8)) * 3
    ).clip(0, 20)

    # 4. Parental education average
    if "mother_education" in df.columns and "father_education" in df.columns:
        df["parental_edu_avg"] = (
            df["mother_education"] + df["father_education"]
        ) / 2.0

    # 5. Digital engagement
    if "login_frequency" in df.columns and "resource_access" in df.columns:
        raw = df["login_frequency"] * df["resource_access"]
        df["digital_engagement"] = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)

    # 6. Study efficiency
    if "study_time" in df.columns and "absences" in df.columns:
        df["study_efficiency"] = df["study_time"] / (df["absences"] + 1.0)

    # 7. Social balance index
    social_cols = {"go_out": 1, "free_time": 1, "dalc": -1, "walc": -1}
    if all(c in df.columns for c in social_cols):
        df["social_index"] = sum(
            df[col] * sign for col, sign in social_cols.items()
        )

    # 8. Behavioural composite
    df["behavioural_composite"] = (
        _safe_col(df, "attendance_rate", 0)
        * _safe_col(df, "assignment_submission_rate", 0)
        * _safe_col(df, "engagement_score", 0)
    )

    # 9. Polynomial terms for period grades
    if "G1" in df.columns:
        df["G1_sq"] = df["G1"] ** 2
    if "G2" in df.columns:
        df["G2_sq"] = df["G2"] ** 2

    return df


def remove_low_variance_features(
    df: pd.DataFrame,
    threshold: float = 0.01,
    exclude: list[str] | None = None,
) -> pd.DataFrame:
    """Drop columns whose normalised variance is below threshold."""
    exclude = exclude or []
    cols_to_check = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in exclude
    ]
    variances = df[cols_to_check].var()
    low_var = variances[variances < threshold].index.tolist()
    if low_var:
        df = df.drop(columns=low_var)
    return df


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_col(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    """Return column or a constant Series if column missing."""
    if col in df.columns:
        return df[col].fillna(default)
    return pd.Series(default, index=df.index)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper: add composites then drop near-zero-variance cols."""
    df = add_composite_features(df)
    return df
