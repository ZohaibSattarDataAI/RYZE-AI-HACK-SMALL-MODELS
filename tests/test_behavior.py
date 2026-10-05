"""Tests for modules.behavior."""

import numpy as np
import pandas as pd

from modules.behavior import analyze_behavior


def _make_df():
    rng = np.random.default_rng(7)
    rows = []
    for month in ["2026-01", "2026-02", "2026-03"]:
        for _ in range(15):
            day = int(rng.integers(1, 28))
            rows.append({
                "date": pd.Timestamp(f"{month}-{day:02d}"),
                "amount": float(rng.normal(2000, 400)),
                "merchant": "Foodpanda",
                "category": "",
                "description": "",
            })
    for month in ["2026-01", "2026-02", "2026-03", "2026-04"]:
        rows.append({
            "date": pd.Timestamp(f"{month}-05"),
            "amount": 25000.0,
            "merchant": "Landlord",
            "category": "Rent",
            "description": "",
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def test_monthly_stats_present():
    report = analyze_behavior(_make_df())
    assert len(report.monthly_stats) >= 3
    months = [m.month for m in report.monthly_stats]
    assert "2026-01" in months


def test_recurring_expense_detected():
    report = analyze_behavior(_make_df())
    merchants = [r.merchant for r in report.recurring]
    assert "Landlord" in merchants


def test_risk_score_bounded():
    report = analyze_behavior(_make_df())
    assert 0 <= report.behavior_risk_score <= 100


def test_empty_input_returns_empty_report():
    report = analyze_behavior(pd.DataFrame())
    assert report.behavior_risk_score == 0