"""
Phase 3 — Spending & behavior analysis.

Computes:
- Monthly spending totals and trends
- Category-wise breakdown (if category column is populated)
- Merchant-wise breakdown (used when category is empty)
- Recurring expense detection (subscriptions, rent, EMI)
- Current-month deviation vs rolling baseline
- Top spending spikes (single-transaction outliers within a category)

No ML. Pure pandas / numpy. Runs instantly on 10k+ rows.

Output is a dictionary of plain Python values so it can be rendered
directly in Streamlit and passed to the fusion engine.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASELINE_MONTHS = 3              # months used to compute "normal" spending
SPIKE_ZSCORE = 2.0               # z-score threshold for a monthly spike
RECURRING_AMOUNT_TOL = 0.10      # ±10% amount tolerance for recurrence
RECURRING_DAY_TOL = 4            # ±4 days tolerance for recurrence
MIN_RECURRING_OCCURRENCES = 3    # need at least this many to call it recurring
BEHAVIOR_RISK_MAX = 100


# ---------------------------------------------------------------------------
# Data classes for structured output
# ---------------------------------------------------------------------------

@dataclass
class MonthlyStat:
    month: str            # "2026-01"
    total: float
    txn_count: int
    avg_txn: float


@dataclass
class RecurringExpense:
    merchant: str
    avg_amount: float
    occurrences: int
    approx_day_of_month: int
    approx_interval_days: int


@dataclass
class Deviation:
    current_month: str
    current_total: float
    baseline_mean: float
    baseline_std: float
    deviation_pct: float
    zscore: float
    is_significant: bool
    direction: str        # "above" | "below" | "normal"


@dataclass
class BehaviorReport:
    monthly_stats: list[MonthlyStat] = field(default_factory=list)
    category_totals: dict[str, float] = field(default_factory=dict)
    merchant_totals: dict[str, float] = field(default_factory=dict)
    recurring: list[RecurringExpense] = field(default_factory=list)
    deviation: Optional[Deviation] = None
    top_spikes: list[dict] = field(default_factory=list)
    behavior_risk_score: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Core analytics
# ---------------------------------------------------------------------------

def _monthly_stats(df: pd.DataFrame) -> list[MonthlyStat]:
    g = (
        df.assign(month=df["date"].dt.to_period("M").astype(str))
        .groupby("month")
        .agg(total=("amount", "sum"), txn_count=("amount", "size"),
             avg_txn=("amount", "mean"))
        .reset_index()
        .sort_values("month")
    )
    return [
        MonthlyStat(
            month=row["month"],
            total=float(row["total"]),
            txn_count=int(row["txn_count"]),
            avg_txn=float(row["avg_txn"]),
        )
        for _, row in g.iterrows()
    ]


def _category_or_merchant_totals(df: pd.DataFrame) -> tuple[dict, dict]:
    cat = (
        df[df["category"].astype(str).str.len() > 0]
        .groupby("category")["amount"].sum()
        .sort_values(ascending=False)
        .to_dict()
    )
    merch = (
        df.groupby("merchant")["amount"].sum()
        .sort_values(ascending=False)
        .head(15)
        .to_dict()
    )
    return (
        {k: float(v) for k, v in cat.items()},
        {k: float(v) for k, v in merch.items()},
    )


def _detect_recurring(df: pd.DataFrame) -> list[RecurringExpense]:
    """
    Detect recurring expenses: same merchant, similar amount, roughly
    evenly spaced in time. Handles rent, subscriptions, EMIs.
    """
    out: list[RecurringExpense] = []
    if df.empty:
        return out

    for merchant, group in df.groupby("merchant"):
        if len(group) < MIN_RECURRING_OCCURRENCES:
            continue
        g = group.sort_values("date").reset_index(drop=True)
        amounts = g["amount"].values
        dates = pd.to_datetime(g["date"]).values

        # Check amount stability
        med = float(np.median(amounts))
        if med <= 0:
            continue
        within = np.abs(amounts - med) / med <= RECURRING_AMOUNT_TOL
        if within.sum() < MIN_RECURRING_OCCURRENCES:
            continue

        # Check interval regularity
        intervals = np.diff(dates[within].astype("datetime64[D]").astype(int))
        if len(intervals) < 2:
            continue
        median_interval = float(np.median(intervals))
        # Accept intervals roughly monthly (25-35 days) or weekly (6-8 days)
        is_monthly = 25 <= median_interval <= 35
        is_weekly = 6 <= median_interval <= 8
        if not (is_monthly or is_weekly):
            continue

        approx_day = int(pd.Timestamp(dates[within][0]).day)
        out.append(RecurringExpense(
            merchant=str(merchant),
            avg_amount=med,
            occurrences=int(within.sum()),
            approx_day_of_month=approx_day,
            approx_interval_days=int(round(median_interval)),
        ))

    return sorted(out, key=lambda r: -r.avg_amount)


def _current_month_deviation(
    monthly: list[MonthlyStat],
) -> Optional[Deviation]:
    """Compare the most recent month against a rolling baseline."""
    if len(monthly) < 2:
        return None

    # Latest month vs. the BASELINE_MONTHS before it
    latest = monthly[-1]
    baseline = monthly[-(BASELINE_MONTHS + 1):-1]
    if not baseline:
        baseline = monthly[:-1]

    baseline_vals = np.array([m.total for m in baseline], dtype=float)
    mean = float(baseline_vals.mean())
    std = float(baseline_vals.std(ddof=0)) if len(baseline_vals) > 1 else 0.0

    if mean <= 0:
        return None

    dev_pct = (latest.total - mean) / mean * 100.0
    z = (latest.total - mean) / std if std > 0 else 0.0
    is_sig = abs(z) >= SPIKE_ZSCORE or abs(dev_pct) >= 40.0
    direction = "above" if dev_pct > 0 else ("below" if dev_pct < 0 else "normal")

    return Deviation(
        current_month=latest.month,
        current_total=latest.total,
        baseline_mean=mean,
        baseline_std=std,
        deviation_pct=dev_pct,
        zscore=float(z),
        is_significant=bool(is_sig),
        direction=direction,
    )


def _top_spikes(df: pd.DataFrame, k: int = 5) -> list[dict]:
    """
    Find transactions that are far above the median for their merchant
    (or globally if merchant has < 3 txns).
    """
    rows = []
    global_med = float(df["amount"].median()) if not df.empty else 0.0
    for _, r in df.iterrows():
        merchant_rows = df[df["merchant"] == r["merchant"]]["amount"]
        med = float(merchant_rows.median()) if len(merchant_rows) >= 3 else global_med
        if med <= 0:
            continue
        ratio = r["amount"] / med
        if ratio >= 3.0:
            rows.append({
                "date": str(pd.Timestamp(r["date"]).date()),
                "amount": float(r["amount"]),
                "merchant": str(r["merchant"]),
                "ratio_vs_median": round(ratio, 2),
            })
    rows.sort(key=lambda x: -x["ratio_vs_median"])
    return rows[:k]


# ---------------------------------------------------------------------------
# Risk score
# ---------------------------------------------------------------------------

def _behavior_risk(
    monthly: list[MonthlyStat],
    deviation: Optional[Deviation],
    recurring: list[RecurringExpense],
    top_spikes: list[dict],
) -> tuple[int, list[str]]:
    """
    Transparent 0-100 behavior risk score. Each contribution is documented
    so the AI Explanation module (Phase 6) can reference it directly.
    """
    score = 0
    notes: list[str] = []

    if deviation:
        if deviation.is_significant and deviation.direction == "above":
            # Scale deviation: 40% -> +30, 100% -> +60, capped
            contrib = int(min(60, max(0, (abs(deviation.deviation_pct) - 40) * 0.6 + 30)))
            score += contrib
            notes.append(
                f"Current month spending is {deviation.deviation_pct:+.0f}% "
                f"vs {len(monthly)-1}-month baseline (+{contrib})"
            )
        elif deviation.is_significant and deviation.direction == "below":
            notes.append(
                f"Spending is {deviation.deviation_pct:+.0f}% below baseline "
                f"(no risk contribution)"
            )

    # Spikes contribute
    if top_spikes:
        spike_contrib = min(30, 8 * len(top_spikes))
        score += spike_contrib
        notes.append(f"{len(top_spikes)} transaction(s) ≥3× merchant median (+{spike_contrib})")

    # Recurring expenses are informational only, not risky
    if recurring:
        notes.append(f"{len(recurring)} recurring expense(s) detected (informational)")

    return int(min(BEHAVIOR_RISK_MAX, score)), notes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_behavior(df: pd.DataFrame) -> BehaviorReport:
    """Run the full Phase 3 pipeline on a cleaned transaction DataFrame."""
    if df.empty:
        return BehaviorReport()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    monthly = _monthly_stats(df)
    cat_totals, merch_totals = _category_or_merchant_totals(df)
    recurring = _detect_recurring(df)
    deviation = _current_month_deviation(monthly)
    spikes = _top_spikes(df)

    score, notes = _behavior_risk(monthly, deviation, recurring, spikes)

    return BehaviorReport(
        monthly_stats=monthly,
        category_totals=cat_totals,
        merchant_totals=merch_totals,
        recurring=recurring,
        deviation=deviation,
        top_spikes=spikes,
        behavior_risk_score=score,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Self-test (python -m modules.behavior)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(7)

    # 4 months of normal spending around Rs 30,000/month
    rows = []
    for month in ["2026-01", "2026-02", "2026-03"]:
        for _ in range(15):
            day = int(rng.integers(1, 28))
            rows.append({
                "date": pd.Timestamp(f"{month}-{day:02d}"),
                "amount": float(rng.normal(2000, 400)),
                "merchant": str(rng.choice(["Foodpanda", "Careem", "K-Electric"])),
                "category": "",
                "description": "",
            })
    # Recurring rent
    for month in ["2026-01", "2026-02", "2026-03", "2026-04"]:
        rows.append({
            "date": pd.Timestamp(f"{month}-05"),
            "amount": 25000.0,
            "merchant": "Landlord",
            "category": "Rent",
            "description": "monthly rent",
        })
    # Current month (April) is a spike: +50k extra
    for _ in range(10):
        rows.append({
            "date": pd.Timestamp("2026-04-15"),
            "amount": 5000.0,
            "merchant": "Amazon",
            "category": "Shopping",
            "description": "",
        })

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    report = analyze_behavior(df)

    print("Monthly spending:")
    for m in report.monthly_stats:
        print(f"  {m.month}: Rs {m.total:,.0f}  ({m.txn_count} txns)")

    print("\nRecurring:")
    for r in report.recurring:
        print(f"  {r.merchant}: Rs {r.avg_amount:,.0f} every ~{r.approx_interval_days}d")

    if report.deviation:
        d = report.deviation
        print(f"\nDeviation ({d.current_month}): {d.deviation_pct:+.1f}%  "
              f"z={d.zscore:.2f}  significant={d.is_significant}")

    print(f"\nBehavior risk score: {report.behavior_risk_score}")
    for n in report.notes:
        print(f"  - {n}")