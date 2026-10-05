"""
FinGuard AI — Privacy-first financial safety dashboard.

Streamlit app. Runs 100% locally. No cloud AI. No outbound calls.

Layout
------
1. Header + privacy badge
2. Sidebar — CSV upload, sample download/load, message input
3. Overall Financial Risk (big metric)
4. Scam Scanner
5. Transaction Analysis
6. Spending Behavior
7. Risk Alerts
8. AI Explanation
9. Privacy Proof
10. Safety footer

Explanation engine note
-----------------------
The dashboard always uses the deterministic template engine for
guaranteed reproducibility and zero model-load latency. An optional
local LLM path (Qwen2.5-0.5B GGUF) is present in modules/explain.py
and activates on AVX2-capable CPUs; the sidebar simply reports
whether the weights are present.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import streamlit as st

# Local modules — no cloud, no network
from modules.ingest import load_transactions, summarize
from modules.anomaly import (
    detect_anomalies, top_anomalies,
    module_risk_score as anomaly_risk,
)
from modules.behavior import analyze_behavior
from modules.scam import scan_message, module_risk_score as scam_risk
from modules.fusion import fuse_risk
from modules.explain import explain_risk, model_info

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FinGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .fg-badge {
        display: inline-block;
        padding: 4px 10px;
        margin-right: 6px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .fg-green  { background: #e7f7ed; color: #1a7f37; border: 1px solid #b6e2c6; }
    .fg-blue   { background: #e8f1fd; color: #1f5dbf; border: 1px solid #bcd4f6; }
    .fg-amber  { background: #fff4e0; color: #925b00; border: 1px solid #ffd699; }
    .fg-red    { background: #fdecec; color: #a01010; border: 1px solid #f7b6b6; }
    .fg-muted  { color: #666; font-size: 0.85rem; }
    .fg-footer {
        margin-top: 40px; padding-top: 16px;
        border-top: 1px solid #e6e6e6;
        font-size: 0.85rem; color: #666;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sample data generator (deterministic, used by download + load buttons)
# ---------------------------------------------------------------------------

def _generate_sample_transactions() -> pd.DataFrame:
    """Return a deterministic synthetic transaction DataFrame (67 rows)."""
    rng = np.random.default_rng(42)
    rows = []
    for month in ["2026-01", "2026-02", "2026-03"]:
        for _ in range(20):
            day = int(rng.integers(1, 28))
            rows.append({
                "date": f"{month}-{day:02d}",
                "amount": round(float(rng.normal(2000, 400)), 2),
                "merchant": str(rng.choice(
                    ["Foodpanda", "Careem", "K-Electric", "Daraz"]
                )),
                "category": str(rng.choice(
                    ["Food", "Transport", "Utilities", "Shopping"]
                )),
                "description": "routine transaction",
            })
    for month in ["2026-01", "2026-02", "2026-03", "2026-04"]:
        rows.append({
            "date": f"{month}-05",
            "amount": 25000.0,
            "merchant": "Landlord",
            "category": "Rent",
            "description": "monthly rent",
        })
    for day, amt in [("15", 120000.0), ("16", 85000.0), ("18", 95000.0)]:
        rows.append({
            "date": f"2026-04-{day}",
            "amount": amt,
            "merchant": "Unknown Crypto Exchange",
            "category": "Crypto",
            "description": "suspicious large transfer",
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "df_clean": None,
        "df_scored": None,
        "behavior_report": None,
        "scam_results": [],
        "scam_lines": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

col_title, col_badges = st.columns([3, 2])

with col_title:
    st.title("🛡️ FinGuard AI")
    st.markdown(
        '<span class="fg-muted">Your money. Your device. Your privacy.</span>',
        unsafe_allow_html=True,
    )

with col_badges:
    st.markdown(
        '<div style="text-align:right; padding-top:12px;">'
        '<span class="fg-badge fg-green">● Internet: OFF</span>'
        '<span class="fg-badge fg-green">Cloud calls: 0</span>'
        '<span class="fg-badge fg-blue">On-device</span>'
        '</div>',
        unsafe_allow_html=True,
    )

st.divider()


# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Inputs")

    # ---------- Transactions CSV ----------
    st.subheader("Transactions (CSV)")
    sample_df = _generate_sample_transactions()
    sample_csv_bytes = sample_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download sample transactions (CSV)",
        data=sample_csv_bytes,
        file_name="finguard_sample_transactions.csv",
        mime="text/csv",
        use_container_width=True,
        help="Deterministic 67-row demo CSV. Reproducible across machines.",
    )

    txn_file = st.file_uploader(
        "Or upload your own bank CSV",
        type=["csv"],
        key="txn_upload",
        help="Columns auto-detected: date, amount, merchant/narration, category",
    )

    if st.button("📊 Load sample transactions", use_container_width=True):
        try:
            df = load_transactions(sample_df)
            st.session_state["df_clean"] = df
            st.session_state["df_scored"] = None
            st.session_state["behavior_report"] = None
            st.success(f"Loaded {len(df)} transactions")
        except Exception as e:
            st.error(f"Sample load failed: {e}")

    if txn_file is not None:
        try:
            df = load_transactions(txn_file)
            st.session_state["df_clean"] = df
            st.session_state["df_scored"] = None
            st.session_state["behavior_report"] = None
            st.success(f"Loaded {len(df)} transactions from file")
        except Exception as e:
            st.error(f"Could not load CSV: {e}")

    if st.session_state["df_clean"] is not None:
        s = summarize(st.session_state["df_clean"])
        st.caption(
            f"Rows: {s['rows']} · "
            f"{s['date_min']} → {s['date_max']} · "
            f"avg Rs {s['amount_mean']:,.0f}"
        )

    st.divider()

    # ---------- Scam messages ----------
    st.subheader("Scam messages")
    messages_text = st.text_area(
        "Paste one or more SMS (one per line)",
        height=140,
        placeholder=(
            "Dear customer, your account has been blocked. "
            "Share your OTP now."
        ),
    )
    scan_clicked = st.button("🔍 Scan messages", use_container_width=True)

    st.divider()

    # ---------- Explanation engine (informational only) ----------
    st.subheader("Explanation engine")
    st.caption("Deterministic template engine (always on-device).")
    info = model_info()
    if info["present"]:
        st.caption(
            f"Optional local LLM ready: {info['parameters']} params · "
            f"{info['size_mb']} MB — activates on AVX2-capable CPUs."
        )
    else:
        st.caption("Optional LLM weights not found — template mode only.")


# ---------------------------------------------------------------------------
# Handle message scanning
# ---------------------------------------------------------------------------

if scan_clicked and messages_text.strip():
    lines = [ln.strip() for ln in messages_text.splitlines() if ln.strip()]
    st.session_state["scam_lines"] = lines
    st.session_state["scam_results"] = [scan_message(ln) for ln in lines]


# ---------------------------------------------------------------------------
# Compute scores
# ---------------------------------------------------------------------------

df_clean = st.session_state["df_clean"]

# Anomaly
if df_clean is not None and not df_clean.empty:
    cached = st.session_state["df_scored"]
    if cached is None or len(cached) != len(df_clean):
        st.session_state["df_scored"] = detect_anomalies(df_clean)
    df_scored = st.session_state["df_scored"]
    txn_score = anomaly_risk(df_scored)
else:
    df_scored = None
    txn_score = 0

# Behavior
if df_clean is not None and not df_clean.empty:
    if st.session_state["behavior_report"] is None:
        st.session_state["behavior_report"] = analyze_behavior(df_clean)
    beh_report = st.session_state["behavior_report"]
    beh_score = beh_report.behavior_risk_score
else:
    beh_report = None
    beh_score = 0

# Scam
scam_results = st.session_state["scam_results"]
scam_score = scam_risk(scam_results) if scam_results else 0

# Fusion
fusion = fuse_risk(scam_score, txn_score, beh_score)

# Explanation — always template for deterministic, zero-latency output.
t0 = time.perf_counter()
explanation = explain_risk(fusion, prefer_llm=False)
explain_ms = (time.perf_counter() - t0) * 1000


# ---------------------------------------------------------------------------
# Section 1 — Overall Financial Risk
# ---------------------------------------------------------------------------

st.subheader("1. Overall Financial Risk")

level_color = {
    "LOW":    ("#1a7f37", "#e7f7ed"),
    "MEDIUM": ("#925b00", "#fff4e0"),
    "HIGH":   ("#a01010", "#fdecec"),
}
fg, bg = level_color[fusion.risk_level]

c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])

with c1:
    st.markdown(
        f"""
        <div style="background:{bg}; padding:20px; border-radius:12px; text-align:center;">
            <div style="font-size:52px; font-weight:700; color:{fg}; line-height:1;">
                {fusion.overall_score}<span style="font-size:22px;">/100</span>
            </div>
            <div style="font-size:18px; font-weight:600; color:{fg}; margin-top:6px;">
                {fusion.risk_level}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with c2:
    st.metric("Scam risk", f"{scam_score}/100")
with c3:
    st.metric("Transaction risk", f"{txn_score}/100")
with c4:
    st.metric("Behavior risk", f"{beh_score}/100")

with st.expander("How the score was computed"):
    st.markdown("**Weighted fusion (transparent, no black box):**")
    for m in fusion.modules:
        st.markdown(
            f"- **{m.name}** · score {m.score} × weight {m.weight:.2f} "
            f"= **+{m.contribution:.1f}** — _{m.reason}_"
        )
    st.caption(
        "Weights are configurable in `config/fusion_weights.yaml`. "
        "Risk levels: 0–39 LOW, 40–69 MEDIUM, 70–100 HIGH."
    )

st.divider()


# ---------------------------------------------------------------------------
# Section 2 — Scam Scanner
# ---------------------------------------------------------------------------

st.subheader("2. Scam Scanner")

if not scam_results:
    st.info("Paste one or more SMS in the sidebar and click **Scan messages**.")
else:
    scam_lines = st.session_state.get("scam_lines", [])
    rows = []
    for msg, r in zip(scam_lines, scam_results):
        rows.append({
            "Label": r.label.upper(),
            "Score": r.score,
            "Signals": ", ".join(r.signals) or "—",
            "Message": msg[:80] + ("…" if len(msg) > 80 else ""),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    worst = max(scam_results, key=lambda r: r.score)
    with st.expander("Why this verdict?"):
        st.markdown(f"**Signals detected:** {', '.join(worst.signals) or 'none'}")
        st.markdown(
            f"**Interpretation:** A score of {worst.score}/100 with rule signals "
            f"like `{', '.join(worst.signals[:3]) or 'none'}`. "
            "This is a **potential risk** signal, not a definitive verdict. "
            "Always verify with your bank."
        )

st.divider()


# ---------------------------------------------------------------------------
# Section 3 — Transaction Analysis
# ---------------------------------------------------------------------------

st.subheader("3. Transaction Analysis")

if df_scored is None or df_scored.empty:
    st.info("Upload a transaction CSV or click **Load sample transactions**.")
else:
    top = top_anomalies(df_scored, k=5)
    top_display = top.copy()
    top_display.columns = ["Date", "Amount (Rs)", "Merchant", "Anomaly score", "Flagged"]
    st.dataframe(top_display, use_container_width=True, hide_index=True)

    n_flagged = int(df_scored["is_anomaly"].sum())
    st.caption(
        f"Method: Isolation Forest · "
        f"{n_flagged} transaction(s) flagged · "
        f"{len(df_scored)} total"
    )

st.divider()


# ---------------------------------------------------------------------------
# Section 4 — Spending Behavior
# ---------------------------------------------------------------------------

st.subheader("4. Spending Behavior")

if beh_report is None:
    st.info("Upload transactions to see spending trends.")
else:
    if beh_report.monthly_stats:
        monthly_df = pd.DataFrame([{
            "Month": m.month,
            "Total (Rs)": f"{m.total:,.0f}",
            "Txns": m.txn_count,
            "Avg (Rs)": f"{m.avg_txn:,.0f}",
        } for m in beh_report.monthly_stats])
        st.dataframe(monthly_df, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)

    with c1:
        if beh_report.deviation:
            d = beh_report.deviation
            sign = "+" if d.deviation_pct >= 0 else ""
            st.metric(
                f"Current month vs baseline ({d.current_month})",
                f"{sign}{d.deviation_pct:.0f}%",
                delta="significant" if d.is_significant else "normal",
                delta_color="inverse" if d.is_significant and d.direction == "above" else "off",
            )
        else:
            st.metric("Current month vs baseline", "—")

    with c2:
        if beh_report.recurring:
            st.markdown("**Recurring expenses:**")
            for r in beh_report.recurring[:5]:
                st.markdown(
                    f"- {r.merchant} · Rs {r.avg_amount:,.0f} "
                    f"every ~{r.approx_interval_days}d"
                )
        else:
            st.markdown("_No recurring expenses detected._")

st.divider()


# ---------------------------------------------------------------------------
# Section 5 — Risk Alerts
# ---------------------------------------------------------------------------

st.subheader("5. Risk Alerts")

alerts = []
if scam_score >= 70:
    alerts.append(("🔴", "High", "Potential scam message detected"))
elif scam_score >= 40:
    alerts.append(("🟠", "Medium", "Suspicious message patterns detected"))

if txn_score >= 70:
    alerts.append(("🔴", "High", "Unusual transaction activity"))
elif txn_score >= 40:
    alerts.append(("🟠", "Medium", "Some transactions deviate from history"))

if beh_score >= 40:
    alerts.append(("🟠", "Medium", "Current-month spending above baseline"))

if not alerts:
    st.success("No significant alerts. Activity looks consistent with your history.")
else:
    for icon, level, text in alerts:
        st.markdown(f"{icon} **{level}** — {text}")

st.caption("Review recommended. FinGuard does not make financial decisions for you.")

st.divider()


# ---------------------------------------------------------------------------
# Section 6 — AI Explanation
# ---------------------------------------------------------------------------

st.subheader("6. AI Explanation")

st.markdown(f"> {explanation.text}")

colA, colB, colC = st.columns(3)
colA.caption(f"Backend: `{explanation.backend}`")
colB.caption(f"Latency: `{explain_ms:.0f} ms`")
colC.caption(f"Fallback used: `{explanation.fallback_used}`")

if explanation.notes:
    with st.expander("Diagnostics"):
        for n in explanation.notes:
            st.caption(f"• {n}")

st.divider()


# ---------------------------------------------------------------------------
# Section 7 — Privacy Proof
# ---------------------------------------------------------------------------

st.subheader("7. Privacy Proof")

c1, c2, c3 = st.columns(3)
c1.metric("Cloud AI APIs called", "0")
c2.metric("Financial data uploaded", "0 bytes")
c3.metric("Processing location", "This device")

info = model_info()
c1, c2, c3 = st.columns(3)
c1.metric("Local AI models", "2", help="Isolation Forest + Qwen2.5-0.5B (optional)")
c2.metric("Largest model params", "494M", help="Qwen2.5-0.5B-Instruct")
c3.metric("Largest model size", f"{info['size_mb']} MB")

st.markdown(
    """
    <div style="margin-top:10px;">
        <span class="fg-badge fg-green">● No outbound network calls</span>
        <span class="fg-badge fg-green">● Weights cached locally</span>
        <span class="fg-badge fg-green">● Works offline</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()


# ---------------------------------------------------------------------------
# Footer — Safety
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="fg-footer">
        <b>FinGuard AI is a safety and decision-support tool.</b>
        It does not make investment decisions, approve loans, or replace your
        bank. All scores are <i>potential risk</i> indicators, not definitive
        verdicts. Always verify with your financial institution.
        <br><br>
        Everything in this app runs on your device. No financial data leaves
        your computer.
    </div>
    """,
    unsafe_allow_html=True,
)