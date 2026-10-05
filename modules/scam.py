"""
Phase 4 — Scam message detection.

Design
------
Layer 1 (active): rule engine — regex patterns for known scam signals.
Layer 2 (optional): local DistilBERT embedding classifier.

Why embeddings are OFF by default
---------------------------------
Base DistilBERT is not trained for sentence similarity; cosine
spread between classes is too small to be discriminative. Rules
cover every known high-severity signal with high precision. The
embedding path is retained for future fine-tuned use.

Enable later by setting USE_EMBEDDINGS = True.

Model budget: DistilBERT-base-uncased = 66M params (< 500M).
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

USE_EMBEDDINGS = False          # rule layer alone is sufficient
MODEL_NAME = "distilbert-base-uncased"
LOCAL_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "scam_detector",
)

RULE_WEIGHT = 1.0
EMBED_WEIGHT = 0.0              # only used if USE_EMBEDDINGS=True

THRESHOLD_SCAM = 70
THRESHOLD_SUSPICIOUS = 40

# ---------------------------------------------------------------------------
# Layer 1 — Rule engine
# ---------------------------------------------------------------------------

RULES = [
    ("otp_request", 1.00, re.compile(
        r"\b(otp|one[\s-]?time[\s-]?password|pin\s*code|verification\s*code|"
        r"code\s+you\s+received|enter\s+your\s+pin)\b", re.IGNORECASE)),

    ("share_credentials", 1.00, re.compile(
        r"\b(share|send|provide|tell\s+us|confirm|verify|update|enter)\b"
        r".{0,40}"
        r"\b(otp|pin|password|cvv|card\s*number|account\s*number)\b",
        re.IGNORECASE | re.DOTALL)),

    ("kyc_block", 0.85, re.compile(
        r"\b(kyc|know[\s-]?your[\s-]?customer|re[\s-]?kyc)\b.{0,60}"
        r"\b(block|suspend|freeze|expire|update|pending|verify)\b",
        re.IGNORECASE | re.DOTALL)),

    ("account_block", 0.80, re.compile(
        r"\b(account|card|sim)\b.{0,40}"
        r"\b(block(ed)?|suspend(ed)?|freeze|deactivat(ed|e)|hold)\b",
        re.IGNORECASE | re.DOTALL)),

    ("prize_bait", 0.85, re.compile(
        r"\b(you\s+have\s+won|you've\s+won|winner|congratulations|"
        r"lucky\s+draw|prize|lottery|reward\s+of|claim\s+your)\b",
        re.IGNORECASE)),

    ("cashback_bait", 0.65, re.compile(
        r"\b(cashback|refund|reimbursement|reward\s+points)\b.{0,40}"
        r"\b(claim|click|verify|expire|now)\b", re.IGNORECASE | re.DOTALL)),

    ("instant_loan", 0.75, re.compile(
        r"\b(pre[\s-]?approved|instant|guaranteed)\b.{0,30}"
        r"\b(loan|credit|financing)\b", re.IGNORECASE | re.DOTALL)),

    ("urgency", 0.55, re.compile(
        r"\b(urgent(ly)?|immediate(ly)?|within\s+\d+\s+(hours?|minutes?|days?)|"
        r"act\s+now|last\s+warning|final\s+notice|expires?\s+(today|soon)|"
        r"before\s+it'?s\s+too\s+late)\b", re.IGNORECASE)),

    # Shortlinks + suspicious TLDs. Prefix http(s):// is optional because
    # SMS often contains bare domains like "bit.ly/claim" or "hbl-secure.tk".
    ("phishing_link", 0.70, re.compile(
        r"(?:https?://)?"
        r"(?:bit\.ly|tinyurl\.com|t\.co|goo\.gl|ow\.ly|is\.gd|buff\.ly)"
        r"|(?:https?://)?[a-z0-9\-]+\.(?:tk|ml|ga|cf|xyz|top|live|online)\b",
        re.IGNORECASE)),

    ("ip_url", 0.75, re.compile(
        r"(?:https?://)?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", re.IGNORECASE)),

    ("impersonation", 0.60, re.compile(
        r"\b(this\s+is\s+your\s+bank|state\s+bank|hbl|meezan|ubl|"
        r"easypaisa|jazzcash|nayapay|sadapay)\b.{0,60}"
        r"\b(team|support|officer|department)\b", re.IGNORECASE | re.DOTALL)),

    ("payment_request", 0.55, re.compile(
        r"\b(send|transfer|pay|deposit)\b.{0,30}"
        r"\b(rs\.?|pkr|amount|money)\b.{0,30}"
        r"\b(to|at|account|number)\b", re.IGNORECASE | re.DOTALL)),

    ("crypto_bait", 0.40, re.compile(
        r"\b(crypto|bitcoin|usdt|trading\s+signal|double\s+your|"
        r"multiply\s+your|guaranteed\s+returns?)\b", re.IGNORECASE)),
]

SUSPICIOUS_TLD = re.compile(r"\.(tk|ml|ga|cf|xyz|top|live|online)\b", re.IGNORECASE)


def rule_score(message: str) -> tuple:
    """
    Return (score 0-100, list of triggered signal names).

    Aggregation: score = (1 - exp(-total_weight)) * 100.
    This is more calibrated than linear scaling because:
      - A single strong rule (~0.75-0.85) lands at 'suspicious' (50-60)
      - Two strong rules land at 'scam' (70-80)
      - Three or more saturate near 90+
    """
    text = message or ""
    hits: list = []
    total = 0.0
    for name, weight, pattern in RULES:
        if pattern.search(text):
            hits.append(name)
            total += weight
    if "click" in text.lower() and SUSPICIOUS_TLD.search(text):
        hits.append("click_suspicious_tld")
        total += 0.35
    score = int(min(100.0, (1.0 - math.exp(-total)) * 100.0))
    return score, hits


# ---------------------------------------------------------------------------
# Layer 2 — Local embedding classifier (disabled by default)
# ---------------------------------------------------------------------------

PROTOTYPES = {
    "scam": [
        "Dear customer your account has been blocked. Share your OTP immediately.",
        "Congratulations! You have won Rs 500000 in our lucky draw. Click here.",
        "Your KYC is pending. Update now at bit.ly/xyz or your SIM will be blocked.",
        "Get an instant pre-approved loan of Rs 200000. No documents required.",
        "Urgent: your card is suspended. Verify your PIN now.",
        "Double your money in 7 days with our crypto trading signals.",
    ],
    "suspicious": [
        "Your order has been shipped and will arrive on Tuesday.",
        "Reminder: your subscription renews next week for Rs 500.",
        "We tried to reach you regarding your recent transaction.",
        "Limited-time offer: 20% cashback on your next purchase.",
    ],
    "legitimate": [
        "Your account statement for March 2026 is now available in the app.",
        "You have received Rs 5000 from Ali Raza. Ref: 123456.",
        "Your electricity bill for February is Rs 3450. Due 15 March.",
        "Thank you for your purchase at Foodpanda. Order ID 99881.",
        "Your ride with Careem is complete. Fare: Rs 450.",
    ],
}


class _EmbeddingClassifier:
    def __init__(self) -> None:
        self.available = False
        self._model = None
        self._tokenizer = None
        self._proto_embeddings: dict = {}
        self._load_error: Optional[str] = None

    def _load(self) -> None:
        if self.available or self._load_error is not None:
            return
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch  # noqa: F401
            source = LOCAL_MODEL_DIR if os.path.isdir(LOCAL_MODEL_DIR) else MODEL_NAME
            self._tokenizer = AutoTokenizer.from_pretrained(source)
            self._model = AutoModel.from_pretrained(source)
            self._model.eval()
            self._compute_prototypes()
            self.available = True
        except Exception as e:
            self._load_error = f"{type(e).__name__}: {e}"

    def _embed(self, texts: list) -> np.ndarray:
        import torch
        inputs = self._tokenizer(
            texts, padding=True, truncation=True, max_length=128,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = self._model(**inputs)
        # CLS token (index 0) as the sentence representation
        emb = out.last_hidden_state[:, 0, :].cpu().numpy()
        norms = np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-9)
        return emb / norms

    def _compute_prototypes(self) -> None:
        for label, samples in PROTOTYPES.items():
            embs = self._embed(samples)
            mean = embs.mean(axis=0)
            mean /= np.linalg.norm(mean).clip(min=1e-9)
            self._proto_embeddings[label] = mean

    def classify(self, message: str):
        self._load()
        if not self.available:
            return None, None, {"error": self._load_error or "unavailable"}
        emb = self._embed([message])[0]
        sims = {label: float(np.dot(emb, proto))
                for label, proto in self._proto_embeddings.items()}
        best = max(sims, key=sims.get)
        scam_sim = sims.get("scam", 0.0)
        legit_sim = sims.get("legitimate", 0.0)
        spread = max(0.0, scam_sim - legit_sim)
        score = int(np.clip(spread * 500.0, 0.0, 100.0))
        return score, best, {k: round(v, 3) for k, v in sims.items()}


_EMBED_CLASSIFIER = _EmbeddingClassifier()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ScamResult:
    label: str
    score: int
    rule_score: int
    embed_score: Optional[int]
    signals: list = field(default_factory=list)
    embed_details: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)


def scan_message(message: str) -> ScamResult:
    if not message or not message.strip():
        return ScamResult("legitimate", 0, 0, None, [], {}, ["empty message"])

    r_score, signals = rule_score(message)

    if USE_EMBEDDINGS:
        e_score, e_label, e_details = _EMBED_CLASSIFIER.classify(message)
        if e_score is not None:
            final = int(round(RULE_WEIGHT * r_score + EMBED_WEIGHT * e_score))
            notes = [f"embedding_label={e_label}"]
        else:
            final = r_score
            notes = ["embedding layer unavailable — rule-only mode"]
    else:
        e_score, e_details = None, {"skipped": "embeddings disabled"}
        final = r_score
        notes = ["rule-only mode"]

    if final >= THRESHOLD_SCAM:
        label = "scam"
    elif final >= THRESHOLD_SUSPICIOUS:
        label = "suspicious"
    else:
        label = "legitimate"

    return ScamResult(label, final, r_score, e_score, signals, e_details, notes)


def scan_messages(messages: list) -> list:
    return [scan_message(m) for m in messages]


def module_risk_score(results: list) -> int:
    if not results:
        return 0
    return int(np.clip(max(r.score for r in results), 0, 100))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    samples = [
        "Dear customer, your account has been blocked. Share your OTP now.",
        "You have won Rs 500000! Click bit.ly/claim to receive.",
        "Your KYC is pending. Update at http://192.168.1.1 or SIM will be blocked.",
        "Your Careem ride is complete. Fare Rs 450.",
        "Your electricity bill for February is Rs 3450. Due 15 March.",
        "URGENT: verify your PIN immediately at hbl-secure.tk/login",
        "Get an instant pre-approved loan of Rs 200000. No documents.",
    ]
    print(f"{'label':<12} {'score':>5} {'signals'}")
    print("-" * 90)
    for msg in samples:
        r = scan_message(msg)
        sig = ",".join(r.signals[:4]) or "none"
        print(f"{r.label:<12} {r.score:>5}  {sig}")
        print(f"             → {msg[:80]}")
    print()
    print(f"Module risk score: {module_risk_score([scan_message(m) for m in samples])}")