"""
Synthetic Student Performance Dataset Generator
Inspired by the UCI Student Performance Dataset with additional
LMS behavioural features and socio-demographic attributes.
"""

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)


def _clamp(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(arr, lo, hi)


def generate_dataset(n_samples: int = 1000, save: bool = True) -> pd.DataFrame:
    """
    Generate a rich synthetic student performance dataset.

    Parameters
    ----------
    n_samples : int
        Number of student records to generate.
    save : bool
        If True, saves the CSV to data/students.csv.

    Returns
    -------
    pd.DataFrame
        Complete synthetic dataset.
    """
    # ── Demographic features ─────────────────────────────────────────────────
    age = rng.integers(15, 23, size=n_samples)
    gender = rng.choice(["M", "F"], size=n_samples, p=[0.52, 0.48])
    address = rng.choice(["Urban", "Rural"], size=n_samples, p=[0.70, 0.30])
    family_size = rng.choice(["LE3", "GT3"], size=n_samples, p=[0.35, 0.65])
    parent_status = rng.choice(["Together", "Apart"], size=n_samples, p=[0.75, 0.25])

    # Parental education  0=none, 1=primary, 2=5th-9th grade, 3=secondary, 4=higher
    medu = rng.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.03, 0.07, 0.22, 0.36, 0.32])
    fedu = rng.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.04, 0.09, 0.27, 0.34, 0.26])

    # Parental jobs
    jobs = ["teacher", "health", "services", "at_home", "other"]
    mjob = rng.choice(jobs, size=n_samples, p=[0.14, 0.13, 0.19, 0.12, 0.42])
    fjob = rng.choice(jobs, size=n_samples, p=[0.09, 0.10, 0.21, 0.17, 0.43])

    # ── Academic / study features ─────────────────────────────────────────────
    study_time       = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.18, 0.44, 0.26, 0.12])
    past_failures    = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.67, 0.20, 0.09, 0.04])
    extra_support    = rng.choice([0, 1], size=n_samples, p=[0.55, 0.45])
    family_support   = rng.choice([0, 1], size=n_samples, p=[0.40, 0.60])
    paid_classes     = rng.choice([0, 1], size=n_samples, p=[0.53, 0.47])
    activities       = rng.choice([0, 1], size=n_samples, p=[0.52, 0.48])
    nursery          = rng.choice([0, 1], size=n_samples, p=[0.26, 0.74])
    higher_ed_aspire = rng.choice([0, 1], size=n_samples, p=[0.07, 0.93])
    internet_access  = rng.choice([0, 1], size=n_samples, p=[0.21, 0.79])
    romantic         = rng.choice([0, 1], size=n_samples, p=[0.67, 0.33])

    # ── Period grades G1 & G2 (0-20) ─────────────────────────────────────────
    # Base score driven by study time, parental education, fewer failures
    base = (
        9.0
        + study_time * 1.2
        + (medu + fedu) * 0.4
        - past_failures * 2.5
        + higher_ed_aspire * 1.0
        + internet_access * 0.8
        - rng.normal(0, 1.5, n_samples)   # noise
    )
    g1 = _clamp(np.round(base + rng.normal(0, 2, n_samples)).astype(float), 0, 20)
    g2 = _clamp(np.round(g1 + rng.normal(0.3, 1.8, n_samples)).astype(float), 0, 20)

    # ── LMS / Behavioural features ────────────────────────────────────────────
    # Correlated with g1+g2 performance
    perf_factor = (g1 + g2) / 40.0  # 0-1

    login_frequency       = _clamp(
        np.round(perf_factor * 18 + rng.uniform(0, 8, n_samples)), 0, 30
    )
    assignment_submit_rate = _clamp(
        perf_factor * 0.7 + rng.uniform(0.0, 0.35, n_samples), 0, 1
    )
    attendance_rate = _clamp(
        perf_factor * 0.65 + rng.uniform(0.1, 0.40, n_samples), 0, 1
    )
    engagement_score = _clamp(
        perf_factor * 6 + rng.uniform(0, 5, n_samples), 0, 10
    )
    forum_posts      = _clamp(
        np.round(perf_factor * 12 + rng.uniform(0, 8, n_samples)), 0, 25
    )
    resource_access  = _clamp(
        np.round(perf_factor * 30 + rng.uniform(0, 25, n_samples)), 0, 60
    )

    # ── Socio-economic ────────────────────────────────────────────────────────
    travel_time = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.41, 0.33, 0.17, 0.09])
    free_time   = rng.choice([1, 2, 3, 4, 5], size=n_samples)
    go_out      = rng.choice([1, 2, 3, 4, 5], size=n_samples)
    dalc        = rng.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.42, 0.28, 0.16, 0.09, 0.05])
    walc        = rng.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.26, 0.29, 0.22, 0.14, 0.09])
    health      = rng.choice([1, 2, 3, 4, 5], size=n_samples)
    absences    = _clamp(np.round(rng.exponential(3.5, n_samples)), 0, 75)

    # ── Final grade G3 (target) ───────────────────────────────────────────────
    g3_raw = (
        0.40 * g2
        + 0.25 * g1
        + study_time * 0.6
        - past_failures * 1.8
        + attendance_rate * 4.0
        + assignment_submit_rate * 3.0
        + engagement_score * 0.3
        - absences * 0.06
        - dalc * 0.3
        + internet_access * 0.5
        + higher_ed_aspire * 0.6
        + rng.normal(0, 1.2, n_samples)
    )
    g3 = _clamp(np.round(g3_raw).astype(float), 0, 20)

    # ── Grade classification labels ───────────────────────────────────────────
    def grade_label(g: float) -> str:
        if g >= 16:  return "A"
        if g >= 14:  return "B"
        if g >= 12:  return "C"
        if g >= 10:  return "D"
        return "F"

    g3_label  = np.array([grade_label(x) for x in g3])
    pass_fail = (g3 >= 10).astype(int)

    # ── Assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame({
        # Demographic
        "age":              age,
        "gender":           gender,
        "address":          address,
        "family_size":      family_size,
        "parent_status":    parent_status,
        "mother_education": medu,
        "father_education": fedu,
        "mother_job":       mjob,
        "father_job":       fjob,
        # Academic
        "study_time":           study_time,
        "past_failures":        past_failures,
        "extra_support":        extra_support,
        "family_support":       family_support,
        "paid_classes":         paid_classes,
        "activities":           activities,
        "nursery":              nursery,
        "higher_ed_aspiration": higher_ed_aspire,
        "internet_access":      internet_access,
        "romantic":             romantic,
        "travel_time":          travel_time,
        "free_time":            free_time,
        "go_out":               go_out,
        "dalc":                 dalc,
        "walc":                 walc,
        "health":               health,
        "absences":             absences,
        "G1":                   g1,
        "G2":                   g2,
        # LMS / Behavioural
        "login_frequency":           login_frequency,
        "assignment_submission_rate": assignment_submit_rate,
        "attendance_rate":           attendance_rate,
        "engagement_score":          engagement_score,
        "forum_posts":               forum_posts,
        "resource_access":           resource_access,
        # Targets
        "G3":            g3,
        "grade_label":   g3_label,
        "pass_fail":     pass_fail,
    })

    if save:
        out_path = Path(__file__).parent / "students.csv"
        df.to_csv(out_path, index=False)
        print(f"Dataset saved → {out_path}  ({len(df):,} rows × {df.shape[1]} cols)")

    return df


if __name__ == "__main__":
    df = generate_dataset(n_samples=1200)
    print(df.head())
    print(df["grade_label"].value_counts())
    print(df["pass_fail"].value_counts())
