"""
Phase 2 — Transaction anomaly detection.

Uses scikit-learn's Isolation Forest (unsupervised, tree-based) to score
each transaction for how much it deviates from the user's own history.

Features
--------
- log1p(amount)          : amount, log-scaled for skew
- hour_of_day            : time-of-day signal
- day_of_week            : weekday/weekend pattern
- merchant_frequency     : how often this merchant appears
- merchant_is_new        : 1 if merchant seen < NEW_MERCHANT_THRESHOLD times
- rolling_amount_ratio   : amount / rolling median of last N txns
- txn_per_day            : txns on the same calendar day

Output
------
DataFrame with original columns + :
    anomaly_raw   : float (raw IsolationForest score, higher = more anomalous)
    anomaly_score : int 0-100 (min-max normalized)
    is_anomaly    : bool (from IsolationForest.predict, respects contamination)

Design
------
- No network. No cloud. Deterministic given a random_state.
- Falls back to z-score method if the dataset has < MIN_ROWS_FOR_IFOREST rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MIN_ROWS_FOR_IFOREST = 20
CONTAMINATION = 0.05          # expected fraction of anomalies
RANDOM_STATE = 42
NEW_MERCHANT_THRESHOLD = 2    # merchant seen fewer than this many times = new
ROLLING_WINDOW = 10           # window (in txns) for rolling amount ratio

FEATURE_COLUMNS = [
    "log_amount",
    "hour_of_day",
    "day_of_week",
    "merchant_frequency",
    "merchant_is_new",
    "rolling_amount_ratio",
    "txn_per_day",
]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn a cleaned transaction DataFrame into numeric features."""
    if df.empty:
        raise ValueError("Cannot build features from an empty DataFrame.")

    feat = pd.DataFrame(index=df.index)

    # Amount, log-scaled
    feat["log_amount"] = np.log1p(df["amount"].astype(float).clip(lower=0))

    # Time features
    dates = pd.to_datetime(df["date"])
    feat["hour_of_day"] = dates.dt.hour.astype(float)
    feat["day_of_week"] = dates.dt.dayofweek.astype(float)

    # Merchant features
    merchant = df["merchant"].fillna("").astype(str)
    counts = merchant.value_counts()
    feat["merchant_frequency"] = merchant.map(counts).astype(float)
    feat["merchant_is_new"] = (
        feat["merchant_frequency"] < NEW_MERCHANT_THRESHOLD
    ).astype(float)

    # Rolling amount ratio (amount relative to recent median)
    sorted_df = df.sort_values("date")
    rolling_median = (
        sorted_df["amount"]
        .rolling(window=ROLLING_WINDOW, min_periods=1)
        .median()
        .replace(0, np.nan)
    )
    ratio = (sorted_df["amount"] / rolling_median).fillna(1.0)
    feat.loc[sorted_df.index, "rolling_amount_ratio"] = ratio

    # Transactions per day
    day_counts = dates.dt.date.value_counts()
    feat["txn_per_day"] = dates.dt.date.map(day_counts).astype(float)

    return feat


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def _zscore_fallback(feat: pd.DataFrame) -> np.ndarray:
    """Statistical fallback: mean absolute z-score across features."""
    scaler = StandardScaler()
    z = np.abs(scaler.fit_transform(feat.values))
    return z.mean(axis=1)


def _minmax_to_100(raw: np.ndarray) -> np.ndarray:
    """
    Map raw anomaly scores to 0-100 with a stable scale.

    IsolationForest's -decision_function returns values roughly in [-0.5, 0.5].
    Normal points cluster near 0; anomalies are positive and well-separated.
    Min-max preserves that separation (unlike percentile rank which forces
    a uniform spread and produces false positives at the cutoff).
    """
    if len(raw) == 0:
        return np.array([])
    lo, hi = float(raw.min()), float(raw.max())
    if hi - lo < 1e-9:
        return np.full(len(raw), 50.0)
    return np.clip((raw - lo) / (hi - lo) * 100.0, 0.0, 100.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = CONTAMINATION,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Score each transaction for anomaly likelihood.

    Parameters
    ----------
    df : cleaned transaction DataFrame (from modules.ingest)
    contamination : expected fraction of anomalies
    random_state : reproducibility

    Returns
    -------
    df.copy() with columns anomaly_raw, anomaly_score, is_anomaly
    """
    if df.empty:
        raise ValueError("Cannot detect anomalies on an empty DataFrame.")

    out = df.copy().reset_index(drop=True)
    feat = _build_features(out)

    if len(out) < MIN_ROWS_FOR_IFOREST:
        method = "zscore_fallback"
        raw = _zscore_fallback(feat)
        # Fallback: flag top 5% via quantile threshold
        cutoff = np.quantile(raw, 1 - contamination)
        flags = raw >= cutoff
    else:
        method = "isolation_forest"
        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(feat[FEATURE_COLUMNS])
        # Higher = more anomalous
        raw = -model.decision_function(feat[FEATURE_COLUMNS])
        # Let sklearn decide the flag — it respects contamination internally
        flags = model.predict(feat[FEATURE_COLUMNS]) == -1

    scores = _minmax_to_100(raw).astype(int)

    out["anomaly_raw"] = raw
    out["anomaly_score"] = scores
    out["is_anomaly"] = flags

    out.attrs["method"] = method
    out.attrs["contamination"] = contamination
    out.attrs["n_flagged"] = int(out["is_anomaly"].sum())

    return out


def top_anomalies(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """Return the top-k highest-scoring transactions for the dashboard."""
    scored = detect_anomalies(df) if "anomaly_score" not in df.columns else df
    return (
        scored.sort_values("anomaly_score", ascending=False)
        .head(k)[["date", "amount", "merchant", "anomaly_score", "is_anomaly"]]
        .reset_index(drop=True)
    )


def module_risk_score(scored: pd.DataFrame) -> int:
    """
    Collapse per-transaction scores into a single 0-100 module risk.

    Anchored on the flagged transactions: if any is flagged, risk reflects
    the highest score among flagged rows. Otherwise, the top score is a
    soft warning. This avoids a single outlier dominating while still
    reflecting the worst case.
    """
    if scored.empty:
        return 0
    flagged = scored[scored["is_anomaly"]]
    if not flagged.empty:
        return int(np.clip(flagged["anomaly_score"].max(), 0, 100))
    return int(np.clip(scored["anomaly_score"].quantile(0.90) * 0.5, 0, 100))


# ---------------------------------------------------------------------------
# Self-test (python -m modules.anomaly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 60
    normal = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=n, freq="D"),
        "amount": rng.normal(2000, 300, n).clip(500, None),
        "merchant": rng.choice(["Foodpanda", "Careem", "K-Electric"], n),
        "category": "",
        "description": "normal",
    })
    anomalies = pd.DataFrame({
        "date": pd.to_datetime(["2026-02-15", "2026-02-20", "2026-02-25"]),
        "amount": [85000, 120000, 95000],
        "merchant": ["Unknown Crypto Exchange"] * 3,
        "category": "",
        "description": "suspicious",
    })
    df = pd.concat([normal, anomalies], ignore_index=True).sort_values("date")

    scored = detect_anomalies(df)
    print(f"Method: {scored.attrs['method']}")
    print(f"Rows: {len(scored)}   Flagged: {scored.attrs['n_flagged']}")
    print(f"Module risk score: {module_risk_score(scored)}")
    print()
    print(top_anomalies(scored, k=5).to_string(index=False))