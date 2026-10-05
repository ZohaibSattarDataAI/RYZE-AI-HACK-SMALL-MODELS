"""
Phase 1 — Transaction data ingestion and cleaning.

Accepts a user-uploaded CSV (path, bytes, file-like object) or a
pandas DataFrame and returns a normalized DataFrame with columns:

    date       : datetime64[ns]
    amount     : float64   (always positive — sign discarded)
    merchant   : str
    category   : str       (empty string if not provided)
    description: str       (raw narration text)

Design goals
------------
- Deterministic. No ML. No network.
- Tolerant of messy bank exports (currency symbols, commas, mixed dates).
- Fails loudly with a helpful error if required columns are missing.
"""

from __future__ import annotations

import io
import re
from typing import Optional, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column detection
# ---------------------------------------------------------------------------

COLUMN_ALIASES = {
    "date": [
        "date", "transaction date", "txn date", "posted date",
        "value date", "trans date", "datetime",
    ],
    "amount": [
        "amount", "amt", "transaction amount", "debit", "value",
        "txn amount",
    ],
    "merchant": [
        "merchant", "narration", "details", "description",
        "payee", "particulars", "remark",
    ],
    "category": ["category", "type", "txn type", "transaction type"],
}

# Currency symbols + separators we strip before numeric parsing.
_CURRENCY_PATTERN = re.compile(r"[Rs$₨,€£\s]")


def _find_column(df: pd.DataFrame, key: str) -> Optional[str]:
    """Return the first column in df that matches a known alias for `key`."""
    lowered = {c.lower().strip(): c for c in df.columns}
    # Exact match first
    for alias in COLUMN_ALIASES[key]:
        if alias in lowered:
            return lowered[alias]
    # Substring match second
    for col_lower, original in lowered.items():
        for alias in COLUMN_ALIASES[key]:
            if alias in col_lower:
                return original
    return None


# ---------------------------------------------------------------------------
# Field cleaners
# ---------------------------------------------------------------------------

def _clean_amount(series: pd.Series) -> pd.Series:
    """Strip currency symbols and commas, coerce to float, take abs value."""
    cleaned = (
        series.astype(str)
        .str.replace(_CURRENCY_PATTERN, "", regex=True)
        .str.replace(r"^\((.*)\)$", r"-\1", regex=True)  # (100) -> -100
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce").abs()


def _clean_date(series: pd.Series) -> pd.Series:
    """Best-effort datetime parsing across mixed formats (day-first)."""
    return pd.to_datetime(series, errors="coerce", dayfirst=True, format="mixed")


def _clean_text(series: pd.Series) -> pd.Series:
    """Trim whitespace and collapse internal spaces."""
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .replace({"nan": "", "None": ""})
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_transactions(
    source: Union[str, bytes, bytearray, io.IOBase, pd.DataFrame]
) -> pd.DataFrame:
    """
    Load a CSV or DataFrame and return a clean, normalized DataFrame.

    Parameters
    ----------
    source : str | bytes | file-like | DataFrame
        Path to CSV, raw bytes, an open file object, or an existing DataFrame.

    Returns
    -------
    pd.DataFrame with columns: date, amount, merchant, category, description
    """
    if isinstance(source, pd.DataFrame):
        raw = source.copy()
    elif isinstance(source, (bytes, bytearray)):
        raw = pd.read_csv(io.BytesIO(source))
    elif hasattr(source, "read"):
        raw = pd.read_csv(source)
    else:
        raw = pd.read_csv(source)

    return clean_transactions(raw)


def clean_transactions(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a raw transaction DataFrame."""
    if raw is None or raw.empty:
        raise ValueError("Transaction file is empty.")

    date_col = _find_column(raw, "date")
    amount_col = _find_column(raw, "amount")
    merchant_col = _find_column(raw, "merchant")
    category_col = _find_column(raw, "category")

    if not date_col or not amount_col:
        raise ValueError(
            "Could not identify required 'date' and/or 'amount' columns. "
            f"Columns found: {list(raw.columns)}"
        )

    out = pd.DataFrame()
    out["date"] = _clean_date(raw[date_col])
    out["amount"] = _clean_amount(raw[amount_col])
    out["description"] = _clean_text(raw[merchant_col]) if merchant_col else ""
    out["merchant"] = out["description"].str.slice(0, 60)
    out["category"] = _clean_text(raw[category_col]) if category_col else ""

    # Drop rows that failed to parse date or amount
    before = len(out)
    out = out.dropna(subset=["date", "amount"]).reset_index(drop=True)
    dropped = before - len(out)

    # Drop exact duplicates
    out = out.drop_duplicates().reset_index(drop=True)

    # Sort chronologically
    out = out.sort_values("date").reset_index(drop=True)

    # Attach diagnostics as metadata (not used downstream, but useful)
    out.attrs["rows_dropped"] = dropped
    out.attrs["rows_kept"] = len(out)

    if out.empty:
        raise ValueError(
            "No valid rows remained after cleaning. "
            "Check that your 'date' and 'amount' columns are correct."
        )

    return out


def summarize(df: pd.DataFrame) -> dict:
    """Return a small summary dict for the dashboard."""
    if df.empty:
        return {"rows": 0}
    return {
        "rows": int(len(df)),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "amount_min": float(df["amount"].min()),
        "amount_max": float(df["amount"].max()),
        "amount_mean": float(df["amount"].mean()),
        "unique_merchants": int(df["merchant"].nunique()),
        "rows_dropped": int(df.attrs.get("rows_dropped", 0)),
    }


# ---------------------------------------------------------------------------
# Self-test (python -m modules.ingest)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo = pd.DataFrame({
        "Transaction Date": ["01/03/2026", "02/03/2026", "2026-03-03", "bad"],
        "Amount": ["Rs 1,500.00", "250.5", "(999)", "1000"],
        "Narration": ["  Careem  ride ", "Foodpanda", "ATM Withdrawal", "x"],
    })
    cleaned = clean_transactions(demo)
    print(cleaned)
    print()
    print(summarize(cleaned))