"""Tests for modules.anomaly."""

import numpy as np
import pandas as pd

from modules.anomaly import detect_anomalies, top_anomalies, module_risk_score


def _make_df(n_normal=40, anomalies=None):
    rng = np.random.default_rng(42)
    rows = [{
        "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=i),
        "amount": float(rng.normal(2000, 300)),
        "merchant": "Routine",
        "category": "",
        "description": "",
    } for i in range(n_normal)]
    for amt in (anomalies or []):
        rows.append({
            "date": pd.Timestamp("2026-03-15"),
            "amount": amt,
            "merchant": "Unknown",
            "category": "",
            "description": "",
        })
    return pd.DataFrame(rows)


def test_flags_large_transaction():
    df = _make_df(anomalies=[200000.0])
    scored = detect_anomalies(df)
    top = scored.nlargest(1, "anomaly_score")
    assert top["amount"].iloc[0] == 200000.0


def test_deterministic_with_same_seed():
    df = _make_df()
    a = detect_anomalies(df)
    b = detect_anomalies(df)
    assert list(a["anomaly_score"]) == list(b["anomaly_score"])


def test_small_dataset_uses_fallback():
    df = _make_df(n_normal=10)
    scored = detect_anomalies(df)
    assert scored.attrs["method"] == "zscore_fallback"


def test_module_risk_score_bounded():
    df = _make_df(anomalies=[200000.0])
    scored = detect_anomalies(df)
    score = module_risk_score(scored)
    assert 0 <= score <= 100


def test_top_anomalies_returns_k_rows():
    df = _make_df(anomalies=[200000.0, 150000.0])
    scored = detect_anomalies(df)
    top = top_anomalies(scored, k=3)
    assert len(top) == 3