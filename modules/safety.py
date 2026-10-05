"""
Cross-cutting safety guardrails.

Applied to any free-text output produced by FinGuard (LLM explanations,
future voice alerts, exported reports). Two layers:

1. Hard blocklist
   If the text contains any phrase here, the text is rejected entirely.

2. Soft rewrites
   Known risky phrasings are rewritten to safer equivalents before
   delivery. Example: "is a scam" -> "shows strong scam signals".

Design principle
----------------
Financial safety tools must never:
- Give investment advice
- Guarantee outcomes
- Make definitive fraud claims
- Approve or reject loans

This module is the last line of defence. It is intentionally narrow
and deterministic.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

# If any of these appear (case-insensitive), the text is rejected.
HARD_BLOCKLIST = [
    "invest in",
    "you should buy",
    "you should sell",
    "guaranteed returns",
    "guaranteed profit",
    "100% safe",
    "100% accurate",
    "definitely fraud",
    "definitely a scam",
    "is definitely",
    "approve the loan",
    "reject the loan",
    "you must invest",
    "double your money",
]

# Soft rewrites applied first. Order matters: longer patterns first.
SOFT_REWRITES = [
    (r"\bis a scam\b", "shows strong scam signals"),
    (r"\bis fraud\b", "shows strong fraud signals"),
    (r"\bis legitimate\b", "shows no strong risk signals"),
    (r"\bthis is fake\b", "this shows risk indicators"),
    (r"\bdefinitely\b", "likely"),
    (r"\bguaranteed\b", "possible"),
    (r"\b100%\s+accurate\b", "high-confidence"),
    (r"\b100%\s+safe\b", "low-risk"),
]

# Required language the app is expected to use when uncertain.
REQUIRED_PHRASES = [
    "potential risk",
    "unusual activity",
    "review recommended",
    "risk indicators",
]


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SafetyReport:
    text: str              # filtered text (empty if rejected)
    accepted: bool         # False if hard blocklist matched
    modified: bool         # True if any soft rewrite applied
    blocked_phrase: str    # which phrase triggered rejection (if any)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_safety(text: str) -> SafetyReport:
    """
    Filter a candidate output through the safety rules.

    Returns a SafetyReport. If accepted is False, the caller MUST discard
    the text and use a fallback (e.g. the template engine).
    """
    if not text:
        return SafetyReport("", True, False, "")

    working = text

    # Layer 1 — soft rewrites
    modified = False
    for pattern, replacement in SOFT_REWRITES:
        new_text, count = re.subn(pattern, replacement, working, flags=re.IGNORECASE)
        if count > 0:
            working = new_text
            modified = True

    # Layer 2 — hard blocklist
    lowered = working.lower()
    for phrase in HARD_BLOCKLIST:
        if phrase in lowered:
            return SafetyReport(
                text="",
                accepted=False,
                modified=modified,
                blocked_phrase=phrase,
            )

    return SafetyReport(
        text=working,
        accepted=True,
        modified=modified,
        blocked_phrase="",
    )


def is_safe(text: str) -> bool:
    """Convenience: True if the text passes all safety rules."""
    return apply_safety(text).accepted


def uses_required_language(text: str) -> bool:
    """
    True if the text uses at least one of the required hedged phrases.
    Used in tests, not in runtime filtering.
    """
    lowered = text.lower()
    return any(p in lowered for p in REQUIRED_PHRASES)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    samples = [
        "Potential high financial risk detected. Review recommended.",
        "This message is definitely fraud. You should invest elsewhere.",
        "The transaction shows strong fraud signals. Please review.",
        "You should invest in crypto for guaranteed returns.",
        "Unusual activity was detected in your transactions.",
        "The loan is definitely approved. You must invest now.",
    ]
    for s in samples:
        r = apply_safety(s)
        status = "ACCEPTED" if r.accepted else "REJECTED"
        extra = f"(blocked: {r.blocked_phrase})" if r.blocked_phrase else ""
        mod = "(modified)" if r.modified else ""
        print(f"[{status:<8}] {mod:<10} {extra}")
        print(f"   in : {s}")
        print(f"   out: {r.text or '—'}")
        print()