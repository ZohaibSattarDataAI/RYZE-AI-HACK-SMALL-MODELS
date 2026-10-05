"""
FinGuard AI — privacy-first financial safety modules.

All inference runs locally on this device.
No cloud AI APIs. No outbound network calls at runtime.

Modules
-------
ingest    : Phase 1 — CSV/DataFrame ingestion and cleaning
anomaly   : Phase 2 — Isolation Forest transaction anomaly detection
behavior  : Phase 3 — Spending trends and deviation analysis
scam      : Phase 4 — Local DistilBERT scam-message classifier
fusion    : Phase 5 — Transparent weighted risk fusion
explain   : Phase 6 — Local LLM explanation (with template fallback)
"""

__version__ = "0.1.0"
__all__ = [
    "ingest",
    "anomaly",
    "behavior",
    "scam",
    "fusion",
    "explain",
]