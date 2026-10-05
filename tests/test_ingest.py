"""Tests for modules.ingest."""

import pandas as pd
import pytest

from modules.ingest import clean_transactions, load_transactions, summarize


def test_cleans_currency_symbols_and_commas():
    raw = pd.DataFrame({
        "date": ["2026-03-01", "2026-03-02"],
        "amount": ["Rs 1,500.00", "250.5"],
        "merchant": ["Careem", "Foodpanda"],
    })
    out = clean_transactions(raw)
    assert out["amount"].iloc[0] == 1500.0
    assert out["amount"].iloc[1] == 250.5


def test_parentheses_become_positive_amount():
    raw = pd.DataFrame({
        "date": ["2026-03-01"],
        "amount": ["(999)"],
        "merchant": ["x"],
    })
    out = clean_transactions(raw)
    assert out["amount"].iloc[0] == 999.0


def test_invalid_dates_are_dropped():
    raw = pd.DataFrame({
        "date": ["2026-03-01", "not-a-date"],
        "amount": [100, 200],
        "merchant": ["a", "b"],
    })
    out = clean_transactions(raw)
    assert len(out) == 1
    assert out["amount"].iloc[0] == 100


def test_duplicates_are_removed():
    raw = pd.DataFrame({
        "date": ["2026-03-01", "2026-03-01"],
        "amount": [100, 100],
        "merchant": ["a", "a"],
    })
    out = clean_transactions(raw)
    assert len(out) == 1


def test_missing_amount_column_raises():
    raw = pd.DataFrame({"date": ["2026-03-01"], "note": ["x"]})
    with pytest.raises(ValueError):
        clean_transactions(raw)


def test_empty_dataframe_raises():
    with pytest.raises(ValueError):
        clean_transactions(pd.DataFrame())


def test_summarize_returns_expected_keys():
    raw = pd.DataFrame({
        "date": ["2026-03-01", "2026-03-02"],
        "amount": [100, 200],
        "merchant": ["a", "b"],
    })
    out = clean_transactions(raw)
    s = summarize(out)
    assert s["rows"] == 2
    assert "date_min" in s
    assert "amount_mean" in s