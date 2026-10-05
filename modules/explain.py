"""
Phase 6 — Local AI explanation.

Takes a FusionResult and produces a short, plain-language explanation.

Three-layer design
------------------
1. Template layer
   Deterministic sentence composition from structured signals.
   Always available, zero deps, works offline.

2. llama.cpp layer
   Qwen2.5-0.5B-Instruct (494M params) via llama-cpp-python.
   Runs locally on CPU. Used when the GGUF is present.

3. Safety filter
   Post-generation blocklist. Rejects outputs containing investment
   advice, guarantees, or definitive fraud claims.

Model budget: Qwen2.5-0.5B = 494M params (< 500M limit).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "explainer",
)
MODEL_FILE = "qwen1_5-0_5b-chat-q4_k_m.gguf"

MAX_TOKENS = 160
TEMPERATURE = 0.3
N_CTX = 1024

# Post-generation safety blocklist. If the LLM output contains any of
# these, we discard it and fall back to the template.
SAFETY_BLOCKLIST = [
    "invest in", "buy ", "sell ", "guaranteed", "definitely fraud",
    "100% safe", "100% accurate", "you should invest", "approve the loan",
    "reject the loan", "is fraud", "is a scam", "is legitimate",
]

# Phrases the safety filter strips or rewrites (softer enforcement).
SOFT_REWRITES = {
    r"\bis a scam\b": "shows strong scam signals",
    r"\bis fraud\b": "shows strong fraud signals",
    r"\bis legitimate\b": "shows no strong risk signals",
    r"\bdefinitely\b": "likely",
    r"\bguaranteed\b": "possible",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Explanation:
    text: str                 # final explanation shown to user
    backend: str              # "template" | "llama_cpp"
    fallback_used: bool       # True if LLM was unavailable or rejected
    notes: list[str]          # diagnostics (not shown to end user)


# ---------------------------------------------------------------------------
# Layer 1 — Template engine
# ---------------------------------------------------------------------------

def _sentence_for_driver(driver_name: str, score: int, detail: str = "") -> str:
    """Compose one natural sentence for a top driver."""
    if driver_name == "scam":
        if score >= 85:
            return "The message contains multiple high-risk scam signals such as OTP requests, urgency, or phishing links"
        if score >= 70:
            return "The message shows strong scam indicators, including requests for sensitive information"
        if score >= 40:
            return "The message contains some suspicious patterns that could be risky"
        return "The message text appears low-risk"
    if driver_name == "transaction":
        if score >= 85:
            return "One or more transactions differ sharply from your usual activity — larger amounts or unfamiliar merchants"
        if score >= 70:
            return "Some transactions deviate from your normal spending pattern"
        if score >= 40:
            return "A few transactions look slightly unusual for your history"
        return "Your transactions are consistent with your history"
    if driver_name == "behavior":
        if score >= 70:
            return "Your current-month spending is notably above your rolling baseline"
        if score >= 40:
            return "Your spending is somewhat higher than your recent average"
        return "Your spending is within your normal range"
    return detail or f"{driver_name} score is {score}"


def _template_explanation(fusion) -> str:
    """Deterministic explanation built from the fusion result."""
    overall = fusion.overall_score
    level = fusion.risk_level

    # Lead-in
    if level == "HIGH":
        lead = f"Potential high financial risk detected (score {overall}/100)."
    elif level == "MEDIUM":
        lead = f"Some financial risk indicators were detected (score {overall}/100)."
    else:
        lead = f"No significant risk indicators were found (score {overall}/100)."

    # Drivers
    drivers = sorted(fusion.modules, key=lambda m: -m.contribution)
    top = [d for d in drivers if d.contribution >= 10][:2]
    if not top:
        top = drivers[:1]

    reasons = []
    for m in top:
        reasons.append(_sentence_for_driver(m.name, m.score))

    body = ""
    if reasons:
        body = " This is because " + ", and ".join(r[0].lower() + r[1:] for r in reasons) + "."

    # Closing recommendation
    if level == "HIGH":
        tail = " We recommend reviewing the flagged items before proceeding with any payment."
    elif level == "MEDIUM":
        tail = " Please review the flagged items at your convenience."
    else:
        tail = " Continue to monitor your account as usual."

    return lead + body + tail


# ---------------------------------------------------------------------------
# Layer 2 — Local LLM via llama.cpp
# ---------------------------------------------------------------------------

_LLM = None
_LLM_LOAD_ERROR: Optional[str] = None


def _load_llm():
    """Lazy-load the GGUF model. Returns model or None on failure."""
    global _LLM, _LLM_LOAD_ERROR
    if _LLM is not None or _LLM_LOAD_ERROR is not None:
        return _LLM

    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    if not os.path.isfile(model_path):
        _LLM_LOAD_ERROR = f"model file not found at {model_path}"
        return None

    try:
        from llama_cpp import Llama
        _LLM = Llama(
            model_path=model_path,
            n_ctx=N_CTX,
            n_threads=max(1, (os.cpu_count() or 2) - 1),
            verbose=False,
        )
        return _LLM
    except Exception as e:  # noqa: BLE001
        _LLM_LOAD_ERROR = f"{type(e).__name__}: {e}"
        return None


def _build_prompt(fusion) -> str:
    """Structured prompt for the LLM. Keeps output short and safe."""
    drivers_txt = "\n".join(f"- {d}" for d in fusion.drivers) or "- none"
    notes_txt = "\n".join(f"- {n}" for n in fusion.notes) or "- none"

    return (
        "You are a financial safety assistant for a privacy-first app. "
        "Explain the risk to a non-technical user in 2 short sentences. "
        "Do NOT give investment advice. Do NOT say 'this is fraud'. "
        "Use phrases like 'potential risk', 'unusual activity', 'review recommended'.\n\n"
        f"Overall risk: {fusion.overall_score}/100 ({fusion.risk_level})\n"
        f"Top drivers:\n{drivers_txt}\n"
        f"Signals:\n{notes_txt}\n\n"
        "Explanation:"
    )


def _llm_explanation(fusion) -> Optional[str]:
    model = _load_llm()
    if model is None:
        return None
    try:
        out = model(
            _build_prompt(fusion),
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            stop=["\n\n", "User:", "Assistant:"],
        )
        text = out["choices"][0]["text"].strip()
        return text or None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Layer 3 — Safety filter
# ---------------------------------------------------------------------------

def _apply_safety(text: str) -> tuple[str, bool]:
    """
    Returns (filtered_text, was_modified).
    Soft-rewrites known risky phrases; if a hard blocklist term remains,
    returns empty string to signal rejection.
    """
    if not text:
        return text, False

    original = text
    for pattern, replacement in SOFT_REWRITES.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    lowered = text.lower()
    for phrase in SAFETY_BLOCKLIST:
        if phrase in lowered:
            return "", True  # hard reject

    return text, text != original


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def explain_risk(fusion, prefer_llm: bool = False) -> Explanation:
    """
    Generate a plain-language explanation for a FusionResult.

    Parameters
    ----------
    fusion : modules.fusion.FusionResult
    prefer_llm : if True, try the local LLM first; fall back to template.

    Returns
    -------
    Explanation
    """
    notes: list[str] = []
    backend = "template"
    fallback = False
    text = ""

    if prefer_llm:
        llm_text = _llm_explanation(fusion)
        if llm_text:
            filtered, modified = _apply_safety(llm_text)
            if filtered:
                text = filtered
                backend = "llama_cpp"
                if modified:
                    notes.append("safety filter softened output")
            else:
                fallback = True
                notes.append("LLM output rejected by safety filter")
        else:
            fallback = True
            notes.append(
                f"LLM unavailable ({_LLM_LOAD_ERROR or 'unknown'}); using template"
            )

    if not text:
        text = _template_explanation(fusion)
        backend = "template"
        if prefer_llm and not fallback:
            fallback = True
            notes.append("template fallback used")

    # Final safety pass on template output too (defensive)
    filtered, modified = _apply_safety(text)
    if modified and filtered:
        notes.append("post-filter softened template output")
    if filtered:
        text = filtered

    return Explanation(
        text=text,
        backend=backend,
        fallback_used=fallback,
        notes=notes,
    )


def model_info() -> dict:
    """For the Privacy Proof panel."""
    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    present = os.path.isfile(model_path)
    size_mb = round(os.path.getsize(model_path) / 1e6, 1) if present else 0
    return {
        "name": "Qwen2.5-0.5B-Instruct (GGUF Q4_K_M)",
        "parameters": "494M",
        "present": present,
        "size_mb": size_mb,
        "backend_available": _LLM is not None or _LLM_LOAD_ERROR is None,
        "load_error": _LLM_LOAD_ERROR,
    }


# ---------------------------------------------------------------------------
# Self-test (python -m modules.explain)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from modules.fusion import fuse_risk

    scenarios = [
        ("Smishing + mule",  90, 85, 60),
        ("Classic scam",     95, 20, 20),
        ("Silent drain",     10, 90, 75),
        ("All clear",        10, 15, 10),
    ]

    print(f"LLM present: {model_info()['present']}  "
          f"(size {model_info()['size_mb']} MB)")
    print("=" * 90)

    for name, s, t, b in scenarios:
        f = fuse_risk(s, t, b)
        e = explain_risk(f)
        print(f"\n[{name}]  overall={f.overall_score} ({f.risk_level})  "
              f"backend={e.backend}  fallback={e.fallback_used}")
        print(f"  → {e.text}")
        if e.notes:
            print(f"  notes: {e.notes}")