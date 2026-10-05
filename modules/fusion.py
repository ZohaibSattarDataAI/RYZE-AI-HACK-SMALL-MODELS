"""
Phase 5 — Risk Fusion Engine.

Combines the three module scores into a single 0-100 Overall Risk Score
and a coarse Risk Level (LOW / MEDIUM / HIGH).

Design principles
-----------------
- Fully transparent. Weighted linear combination with documented weights.
- Config-driven. Weights and thresholds loaded from config/fusion_weights.yaml
  (falls back to hardcoded defaults if the file is missing).
- Every contribution is returned so the AI Explanation module (Phase 6)
  can quote exactly what drove the score.
- Floor rule. A single high-scoring module cannot be masked by low scores
  in the other modules (e.g. scam=90, txn=0, beh=0 → at least HIGH).
- No model, no network, no latency. Pure arithmetic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Config loading (YAML with safe defaults)
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS = {
    "scam": 0.40,
    "transaction": 0.35,
    "behavior": 0.25,
}

_DEFAULT_THRESHOLDS = {
    "low_max": 39,       # 0-39  -> LOW
    "medium_max": 69,    # 40-69 -> MEDIUM
                         # 70-100 -> HIGH
}

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "fusion_weights.yaml",
)


def _load_config() -> tuple:
    """Try to load YAML config; fall back to defaults on any error."""
    weights = dict(_DEFAULT_WEIGHTS)
    thresholds = dict(_DEFAULT_THRESHOLDS)
    try:
        import yaml  # optional at runtime
        if os.path.isfile(_CONFIG_PATH):
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            weights.update(cfg.get("weights", {}))
            thresholds.update(cfg.get("thresholds", {}))
    except Exception:
        # Any failure → safe defaults. Never crash the app.
        pass
    return weights, thresholds


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ModuleScore:
    name: str
    score: int                      # 0-100
    weight: float                   # 0-1
    contribution: float             # score * weight
    reason: str = ""                # short human-readable reason

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FusionResult:
    overall_score: int
    risk_level: str                 # "LOW" | "MEDIUM" | "HIGH"
    modules: list = field(default_factory=list)
    drivers: list = field(default_factory=list)    # top contributors
    notes: list = field(default_factory=list)
    weights_used: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["modules"] = [m.to_dict() for m in self.modules]
        return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clip(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(max(lo, min(hi, v)))


def _level_for(score: int, thresholds: dict) -> str:
    if score <= thresholds["low_max"]:
        return "LOW"
    if score <= thresholds["medium_max"]:
        return "MEDIUM"
    return "HIGH"


def _reason_for(name: str, score: int) -> str:
    if name == "scam":
        if score >= 70:
            return "High-confidence scam signals detected in message text"
        if score >= 40:
            return "Suspicious message patterns detected"
        return "No strong scam signals"
    if name == "transaction":
        if score >= 70:
            return "One or more transactions deviate strongly from your history"
        if score >= 40:
            return "Some transactions look unusual for your pattern"
        return "Transactions look consistent with your history"
    if name == "behavior":
        if score >= 70:
            return "Current spending is well above your rolling baseline"
        if score >= 40:
            return "Moderate spending deviation from baseline"
        return "Spending is within normal range"
    return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fuse_risk(
    scam_score: int = 0,
    transaction_score: int = 0,
    behavior_score: int = 0,
    *,
    weights: Optional[dict] = None,
    thresholds: Optional[dict] = None,
) -> FusionResult:
    """
    Combine module scores into an overall risk score.

    Parameters
    ----------
    scam_score, transaction_score, behavior_score : int (0-100)
    weights     : optional override {scam, transaction, behavior}
    thresholds  : optional override {low_max, medium_max}

    Returns
    -------
    FusionResult with overall_score, risk_level, per-module contributions,
    top drivers, and a notes list for the explanation module.
    """
    cfg_weights, cfg_thresholds = _load_config()
    w = {**cfg_weights, **(weights or {})}
    t = {**cfg_thresholds, **(thresholds or {})}

    # Normalize weights so they always sum to 1.0 (defensive)
    total_w = sum(w.values())
    if total_w <= 0:
        w = dict(_DEFAULT_WEIGHTS)
        total_w = sum(w.values())
    w = {k: v / total_w for k, v in w.items()}

    scam_s = int(_clip(scam_score))
    txn_s = int(_clip(transaction_score))
    beh_s = int(_clip(behavior_score))

    modules = [
        ModuleScore("scam", scam_s, w["scam"], scam_s * w["scam"],
                    _reason_for("scam", scam_s)),
        ModuleScore("transaction", txn_s, w["transaction"], txn_s * w["transaction"],
                    _reason_for("transaction", txn_s)),
        ModuleScore("behavior", beh_s, w["behavior"], beh_s * w["behavior"],
                    _reason_for("behavior", beh_s)),
    ]

    overall = int(round(sum(m.contribution for m in modules)))
    overall = int(_clip(overall))

    # Floor rule: a single high-scoring module must not be masked by
    # low scores in the other two. Prevents confusing cases like
    # scam=90, txn=0, beh=0 → overall=28 LOW.
    max_module = max(scam_s, txn_s, beh_s)
    if max_module >= 85:
        overall = max(overall, 70)      # force at least HIGH
    elif max_module >= 70:
        overall = max(overall, 40)      # force at least MEDIUM

    level = _level_for(overall, t)

    # Top drivers: any module contributing >= 15 points to the overall score
    drivers = [
        f"{m.name}: {m.score}/100 (weight {m.weight:.2f} → +{m.contribution:.1f})"
        for m in sorted(modules, key=lambda x: -x.contribution)
        if m.contribution >= 15
    ]

    # Human-readable notes for the explanation module
    notes = [m.reason for m in modules if m.reason]

    # Special-case flag: high scam + high transaction is the classic
    # "smishing + money mule" combo and worth calling out.
    if scam_s >= 70 and txn_s >= 70:
        notes.append(
            "Message risk and transaction risk are both elevated — "
            "possible smishing linked to a recent transfer."
        )

    return FusionResult(
        overall_score=overall,
        risk_level=level,
        modules=modules,
        drivers=drivers,
        notes=notes,
        weights_used=w,
    )


def fuse_from_reports(
    scam_results: Optional[list] = None,
    txn_scored_df=None,
    behavior_report=None,
) -> FusionResult:
    """
    Convenience wrapper that takes the actual outputs of Phases 2-4
    and fuses them. Import lazily to avoid circular imports.
    """
    # Scam
    scam_score = 0
    if scam_results:
        from modules.scam import module_risk_score
        scam_score = module_risk_score(scam_results)

    # Transaction
    txn_score = 0
    if txn_scored_df is not None and not txn_scored_df.empty:
        from modules.anomaly import module_risk_score as anomaly_risk
        txn_score = anomaly_risk(txn_scored_df)

    # Behavior
    beh_score = 0
    if behavior_report is not None:
        beh_score = int(getattr(behavior_report, "behavior_risk_score", 0))

    return fuse_risk(scam_score, txn_score, beh_score)


# ---------------------------------------------------------------------------
# Self-test (python -m modules.fusion)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    scenarios = [
        ("All clear",        10,  15,  10),
        ("Mild concern",     45,  50,  35),
        ("Classic scam",     95,  20,  20),
        ("Smishing + mule",  90,  85,  60),
        ("Silent drain",     10,  90,  75),
        ("Floor check",      90,   0,   0),
    ]
    for name, s, t, b in scenarios:
        r = fuse_risk(s, t, b)
        print(f"{name:<20} scam={s:>3} txn={t:>3} beh={b:>3} "
              f"→ overall={r.overall_score:>3}  level={r.risk_level}")
        if r.drivers:
            for d in r.drivers:
                print(f"                     driver: {d}")
        print()