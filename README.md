```markdown
# 🛡️ FinGuard AI

<div align="center">

### 🔒 Privacy-First Financial Safety • 📱 Scam Detection • 💳 Transaction Anomaly • 📊 Spending Behavior • 🧠 Risk Fusion • 🗣️ Local AI Explanation

**A fully local, Streamlit-based AI financial safety agent that analyzes transactions, messages, and spending patterns — entirely on your device. No cloud AI. No data uploads. Works offline.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](#-requirements)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](#-application-features)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](#-transaction-anomaly-detector)
[![Transformers](https://img.shields.io/badge/NLP-DistilBERT-FFD21E)](#-scam-message-scanner)
[![llama.cpp](https://img.shields.io/badge/LLM-llama.cpp-000000)](#-local-ai-explanation)
[![Offline](https://img.shields.io/badge/Offline-100%25-1a7f37)](#-privacy-proof-mode)
[![Cloud%20APIs](https://img.shields.io/badge/Cloud%20AI%20APIs-0-a01010)](#-privacy-proof-mode)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)](#-requirements)

</div>

---

## 📌 Overview

**FinGuard AI** is a privacy-first financial safety agent that runs entirely on your own device. It analyzes a user's financial activity — messages, transactions, and spending behavior — and produces a single, explainable **Overall Financial Risk Score** with a plain-language explanation.

The project was built for the **Ryze AI Hack 2026**, where all AI models must be **≤500M parameters** and must run **fully on-device**, with **zero cloud inference APIs**.

FinGuard AI combines multiple independent modules into one dashboard:

- 📱 **Scam Message Scanner** — detects OTP requests, KYC scams, fake loans, prize baits, phishing links, and urgency threats
- 💳 **Transaction Anomaly Detector** — uses Isolation Forest to flag transactions that deviate from the user's history
- 📊 **Spending Behavior Analyzer** — computes monthly trends, recurring expenses, and current-month deviations
- 🧠 **Risk Fusion Engine** — combines signals with transparent, configurable weights
- 🗣️ **Local AI Explanation** — plain-language summaries in simple English, with a deterministic template engine always available and an optional local LLM path
- 🔒 **Privacy Proof Mode** — a live panel showing zero cloud calls, zero uploads, and on-device processing

> **Important:** FinGuard AI is a **safety and decision-support tool**. It does **not** make investment decisions, approve or reject loans, or replace a bank. All scores are **potential risk** indicators, not definitive verdicts.

---

## ✨ Why This Project?

Traditional financial safety tools fall into one of two categories:

1. **Cloud-based fraud detection** — powerful, but requires uploading sensitive financial data to third-party servers
2. **Rule-based local alerts** — private, but brittle and blind to context

FinGuard AI takes a third path: **local AI with multiple independent signal layers**, fused with transparent math and explained in plain language — without ever sending a byte off the device.

### 🎯 Project Goal

> **Make privacy-first financial safety genuinely useful, measurable, and reproducible on a consumer laptop.**

---

# 🚀 Application Features

## 📱 1. Scam Message Scanner

Analyzes SMS or pasted messages and returns a three-way label:

- **Legitimate**
- **Suspicious**
- **Scam**

### Detection patterns

| Pattern | Signal name | Weight |
|---|---|---|
| OTP / PIN request | `otp_request` | 1.00 |
| Share credentials verb + sensitive noun | `share_credentials` | 1.00 |
| KYC block / re-KYC urgency | `kyc_block` | 0.85 |
| Account / SIM / card blocked | `account_block` | 0.80 |
| Prize / lottery / congratulations bait | `prize_bait` | 0.85 |
| Instant / pre-approved loan bait | `instant_loan` | 0.75 |
| IP-address URL | `ip_url` | 0.75 |
| Phishing shortlink or suspicious TLD | `phishing_link` | 0.70 |
| Cashback / refund bait | `cashback_bait` | 0.65 |
| Impersonation of a bank | `impersonation` | 0.60 |
| Payment request with amount + account | `payment_request` | 0.55 |
| Urgency / threat language | `urgency` | 0.55 |
| Crypto / guaranteed returns bait | `crypto_bait` | 0.40 |

### Scoring

Rule hits are summed and mapped via an exponential saturation curve:

```text
score = (1 - exp(-total_weight)) * 100
```

This gives calibrated behaviour:
- One strong rule (~0.80) → ~55 (Suspicious)
- Two strong rules → ~75 (Scam)
- Three or more → ~90+ (High-confidence Scam)

### Optional embedding layer

A local **DistilBERT-base-uncased** model (**66M parameters**) is present in the code for a second signal layer. It is **disabled by default** because base DistilBERT embeddings lack sentence-similarity training, and empirically the rule layer alone gave stronger precision. The embedding path can be re-enabled by setting `USE_EMBEDDINGS = True` in `modules/scam.py`.

---

## 💳 2. Transaction Anomaly Detector

Analyzes a user-uploaded CSV of transactions and assigns each one an anomaly score (0–100).

### Features used

- `log1p(amount)` — log-scaled amount
- `hour_of_day` — time-of-day signal
- `day_of_week` — weekday/weekend pattern
- `merchant_frequency` — how often this merchant appears
- `merchant_is_new` — first-time merchant flag
- `rolling_amount_ratio` — amount relative to recent median
- `txn_per_day` — transactions on the same day

### Algorithm

**Isolation Forest** (`scikit-learn`, 200 trees, `contamination=0.05`) — an unsupervised, tree-based method that requires no labeled fraud data and produces interpretable anomaly scores.

For datasets with fewer than 20 rows, the module falls back to a **mean absolute z-score** across features.

### Output

Each transaction gets:

```text
anomaly_raw    : float   (raw Isolation Forest score)
anomaly_score  : int      (0–100, min-max normalized)
is_anomaly     : bool     (from IsolationForest.predict)
```

The module risk score is derived from the highest-scoring flagged transaction (or a soft 90th percentile if nothing is flagged).

---

## 📊 3. Spending Behavior Analyzer

Pure statistics. No model. Runs instantly on 10,000+ rows.

### What it computes

- **Monthly spending totals** with transaction counts and averages
- **Category-wise breakdown** (when category column is populated)
- **Merchant-wise totals** (top 15)
- **Recurring expense detection** — same merchant, stable amount, roughly evenly spaced (monthly or weekly)
- **Current-month deviation** — vs a 3-month rolling baseline, with percent deviation and z-score
- **Top spending spikes** — transactions ≥3× the median for their merchant

### Recurring expense rules

```text
- Same merchant, ≥3 occurrences
- Amount within ±10% of median
- Interval ~25–35 days (monthly) or ~6–8 days (weekly, ≥4 occurrences)
```

### Risk contribution

- Current-month deviation >40% and above baseline → up to **+60**
- Each top spike → **+8** (capped at **+30**)
- Recurring expenses → informational only, **no risk contribution**

---

## 🧠 4. Risk Fusion Engine

Combines the three module scores into one **Overall Financial Risk Score (0–100)** and a **Risk Level** (LOW / MEDIUM / HIGH).

### Weighted linear fusion

```text
Overall = (scam × w₁) + (transaction × w₂) + (behavior × w₃)
```

Default weights (configurable in `config/fusion_weights.yaml`):

| Module | Weight | Rationale |
|---|---:|---|
| Scam | 0.40 | Highest weight — OTP/KYC fraud is the most common and highest-impact vector |
| Transaction | 0.35 | Strong behavioural signal, but can be noisy on small datasets |
| Behavior | 0.25 | Contextual; useful but slow-moving |

### Floor rule

If any single module scores **≥85**, the overall risk cannot be LOW.
If any module scores **≥70**, the overall risk cannot be LOW.

This prevents a confusing situation where one module is screaming "100/100" but the overall shows "35 LOW".

### Risk levels

| Range | Level |
|---|---|
| 0–39 | 🟢 LOW |
| 40–69 | 🟠 MEDIUM |
| 70–100 | 🔴 HIGH |

Every contribution is returned as structured data so the explanation module can quote it verbatim.

---

## 🗣️ 5. Local AI Explanation

Produces a short, plain-language explanation of the risk score.

### Two backends

**Template engine (default, always available)**

Deterministic sentence composition from the fusion result. Zero latency, zero model load, fully reproducible. Example output:

> *"Potential high financial risk detected (score 81/100). This is because the message contains multiple high-risk scam signals such as OTP requests, urgency, or phishing links, and one or more transactions differ sharply from your usual activity. We recommend reviewing the flagged items before proceeding with any payment."*

**Local LLM (optional, AVX2 CPUs)**

**Qwen2.5-0.5B-Instruct** (494M params, GGUF Q4_K_M, ~407 MB) loaded via `llama-cpp-python`. Generates free-form explanations in natural English.

### Safety filter

Every LLM output passes through a post-generation filter:

- **Hard blocklist** — output is discarded if it contains "invest in", "guaranteed", "definitely fraud", "approve the loan", etc.
- **Soft rewrites** — "is a scam" → "shows strong scam signals", "definitely" → "likely"

If the filter rejects an output, the app falls back to the template engine.

---

## 🔒 6. Privacy Proof Mode

A dedicated dashboard section that proves FinGuard runs locally.

| Metric | Value |
|---|---|
| Cloud AI APIs called | **0** |
| Financial data uploaded | **0 bytes** |
| Processing location | **This device** |
| Local AI models | **2** (Isolation Forest + optional Qwen2.5-0.5B) |
| Largest model params | **494M** |
| Largest model size | **~407 MB** |

**The entire application remains fully functional with Wi-Fi disabled.**

---

# 🏗️ Architecture

```text
User Financial Data
       │
       ▼
┌─────────────────────────────────────────┐
│                                         │
│  Scam Scanner       (rules + optional   │
│  Transaction Det.    DistilBERT)        │
│  Behavior Analysis  (Isolation Forest   │
│                      + statistics)      │
│                                         │
└──────────────────┬──────────────────────┘
                   │
                   ▼
         Risk Fusion Engine
         (transparent weighted sum)
                   │
                   ▼
         Overall Risk Score + Level
                   │
                   ▼
      Local AI Explanation
      (template | optional local LLM)
                   │
                   ▼
              FinGuard UI
           (Streamlit dashboard)
```

---

# 📁 Project Structure

```text
finguard-ai/
│
├── app.py                     # Streamlit dashboard
├── privacy_proof.py           # Standalone privacy monitor (optional)
├── evaluate.py                # Evaluation / benchmarking script
│
├── README.md
├── requirements.txt
├── requirements-lock.txt
├── .gitignore
│
├── modules/
│   ├── __init__.py
│   ├── ingest.py              # Phase 1 — CSV ingestion & cleaning
│   ├── anomaly.py             # Phase 2 — Isolation Forest detector
│   ├── behavior.py            # Phase 3 — Spending trends & deviations
│   ├── scam.py                # Phase 4 — Scam message rules + optional embeddings
│   ├── fusion.py              # Phase 5 — Transparent risk fusion
│   └── explain.py             # Phase 6 — Template + optional local LLM
│
├── config/
│   └── fusion_weights.yaml    # Configurable weights and thresholds
│
├── models/
│   ├── scam_detector/         # DistilBERT weights (cached on first run)
│   ├── explainer/             # Qwen2.5-0.5B GGUF (optional, downloaded once)
│   └── README.md
│
├── data/
│   └── sample_transactions.csv
│
├── evaluation/
│   ├── evaluate.py
│   └── results/
│       └── baseline.md        # Committed benchmark results
│
├── tests/
│   ├── test_ingest.py
│   ├── test_anomaly.py
│   ├── test_behavior.py
│   ├── test_scam.py
│   └── test_fusion.py
│
└── assets/
    └── privacy_proof.png
```

---

# ⚙️ Requirements

## Software

- Python **3.11+**
- pip
- Internet connection for **first-time model downloads only**
- After models are cached, the application runs **fully offline**

## Hardware

The application runs on CPU. **No GPU required.**

| Component | Minimum | Recommended |
|---|---|---|
| CPU | Dual-core | Quad-core with AVX2 |
| RAM | 4 GB | 8 GB+ |
| Disk | ~1 GB | 2 GB+ |
| GPU | Not required | Not required |

> **Note on LLM backend:** The optional local LLM path (Qwen2.5-0.5B) requires an **AVX2-capable CPU** (Intel i5 8th gen+, AMD Ryzen 2000+). On older CPUs (e.g. Intel Celeron N4500), the app automatically uses the deterministic template engine — no crash, no fallback error.

## Model budget

| Module | Model | Parameters |
|---|---|---:|
| Scam | Rule engine | 0 |
| Scam (optional) | DistilBERT-base-uncased | 66M |
| Transaction | Isolation Forest | 0 |
| Behavior | Statistical | 0 |
| Fusion | Arithmetic | 0 |
| Explanation | Qwen2.5-0.5B-Instruct (optional) | 494M |

**Every model is under the hackathon's 500M parameter limit.**

---

# 🪟 Installation on Windows

## Step 1 — Clone the repository

```bat
git clone https://github.com/YOUR_USERNAME/finguard-ai.git
cd finguard-ai
```

Replace `YOUR_USERNAME` with your GitHub username.

## Step 2 — Create a virtual environment

```bat
python -m venv .venv
.venv\Scripts\activate
```

Verify activation — the prompt should show `(.venv)`.

## Step 3 — Install dependencies

```bat
pip install -r requirements.txt
```

If `torch` installation is heavy, use the CPU-only wheel:

```bat
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Step 4 — (Optional) Download model weights

The app works without any pre-downloaded weights — the rule layer and statistical models run on pure Python. To enable the optional LLM explanation path:

```bat
hf download Qwen/Qwen2.5-0.5B-Instruct-GGUF qwen2.5-0.5b-instruct-q4_k_m.gguf --local-dir models\explainer
```

The DistilBERT weights (for the optional scam embedding layer) download automatically on first `python -m modules.scam` run.

## Step 5 — Start the application

```bat
streamlit run app.py
```

Open the URL shown in the terminal — normally:

```text
http://localhost:8501
```

---

# 🐧 Installation on Linux / macOS

```bash
git clone https://github.com/YOUR_USERNAME/finguard-ai.git
cd finguard-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

---

# 🧪 Running the Application

Once the app is running:

1. Open the browser dashboard.
2. **Sidebar** → upload a bank CSV, or click **Load sample transactions**.
3. **Sidebar** → paste one or more SMS messages.
4. Click **🔍 Scan messages**.
5. Scroll through the dashboard sections:

| Section | Purpose |
|---|---|
| 1. Overall Financial Risk | Big risk score + level |
| 2. Scam Scanner | Per-message labels, scores, and signals |
| 3. Transaction Analysis | Top 5 anomalies with scores |
| 4. Spending Behavior | Monthly table, deviation, recurring expenses |
| 5. Risk Alerts | Plain-language alert cards |
| 6. AI Explanation | Full explanation of the fused score |
| 7. Privacy Proof | Zero cloud calls, on-device, offline |

---

# 📱 How to Use the Scam Scanner

## Step 1 — Paste a message

In the sidebar, paste a message (one per line for multiple):

```text
Dear customer, your account has been blocked. Share your OTP now.
You have won Rs 500000! Click bit.ly/claim to receive.
Your KYC is pending. Update at http://192.168.1.1 or SIM will be blocked.
```

## Step 2 — Scan

Click **🔍 Scan messages**.

## Step 3 — Read the verdict

The **Scam Scanner** section shows each message with its label, score, and signals:

| Label | Score | Signals |
|---|---:|---|
| SCAM | 93 | otp_request, share_credentials, account_block |
| SCAM | 78 | prize_bait, phishing_link |
| SCAM | 90 | kyc_block, account_block, ip_url |

---

# 💳 How to Analyze Transactions

## Step 1 — Prepare a CSV

Required columns (auto-detected by alias):

- `date` (or `transaction date`, `posted date`, …)
- `amount` (or `amt`, `debit`, …)
- `merchant` (or `narration`, `description`, …) — optional
- `category` — optional

## Step 2 — Upload

Use the **Or upload your own bank CSV** uploader in the sidebar, or click **📊 Load sample transactions** to use the built-in 67-row sample.

## Step 3 — Read the analysis

- **Transaction Analysis** shows the top 5 highest-scoring anomalies.
- **Spending Behavior** shows monthly totals, deviation from baseline, and recurring expenses.

---

# 📊 How to Read the Dashboard Sections

| Section | What it shows |
|---|---|
| **1. Overall Financial Risk** | Fused score (0–100) with LOW/MEDIUM/HIGH level. Expandable "How the score was computed" panel shows every module's contribution. |
| **2. Scam Scanner** | Table of scanned messages with label, score, and detected signals. Click "Why this verdict?" to see the worst message's analysis. |
| **3. Transaction Analysis** | Top 5 anomalies with date, amount, merchant, anomaly score, and flag status. |
| **4. Spending Behavior** | Monthly spending table, current-month deviation, and recurring expenses. |
| **5. Risk Alerts** | Plain-language alert cards — High/Medium — for each flagged module. |
| **6. AI Explanation** | Full plain-language explanation of the fused score, with backend, latency, and fallback info. |
| **7. Privacy Proof** | Live proof of local processing — zero cloud calls, zero uploads, on-device. |

---

# 🔍 How the Modules Work Together

```text
┌────────────────────────────────────────────────────────────┐
│                    User Inputs                             │
│  · Transaction CSV (bank statement)                        │
│  · Scam messages (pasted SMS text)                         │
└──────────────────────┬─────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────────┐
│                  Processing Layers                         │
│                                                            │
│  1. Ingest        → clean, normalize, validate             │
│  2. Anomaly       → Isolation Forest per-transaction score │
│  3. Behavior      → statistical trends + deviation         │
│  4. Scam          → rule engine (+ optional DistilBERT)    │
│  5. Fusion        → weighted risk combination              │
│  6. Explanation   → template | optional local LLM          │
└──────────────────────┬─────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────────┐
│                     Outputs                                │
│  · Overall risk score (0–100) with LOW/MEDIUM/HIGH         │
│  · Per-module breakdown with reasons                       │
│  · Plain-language explanation                              │
│  · Privacy proof dashboard                                 │
└────────────────────────────────────────────────────────────┘
```

---

# 🗣️ How to Read the AI Explanation

The **AI Explanation** section shows a plain-language explanation of the fused risk score.

It always displays:

- **Backend** — `template` (always) or `llama_cpp` (when available)
- **Latency** — milliseconds for generating the explanation
- **Fallback used** — `True` if the LLM was unavailable

The template engine runs in **~1 ms**. The LLM path takes **1–3 seconds** on AVX2 CPUs, and much longer on older hardware.

---

# 📊 Benchmarking

Run the evaluation script:

```bat
python evaluate.py
```

The script produces a Markdown report with:

- **Scam module** — accuracy, precision, recall, F1, confusion matrix
- **Transaction anomaly** — detection rate on injected anomalies, false-positive rate
- **System** — inference latency per module, RAM usage, model size, parameter count

Results are written to `evaluation/results/`.

---

# 🧪 Testing

Install `pytest`:

```bat
pip install pytest
```

Run all tests:

```bat
pytest tests/ -v
```

Recommended coverage:

- `test_ingest.py` — currency symbols, mixed dates, missing columns, empty CSV
- `test_anomaly.py` — synthetic anomalies are flagged, determinism, latency
- `test_behavior.py` — recurring detection, deviation calculation, spike detection
- `test_scam.py` — OTP, KYC, prize, phishing, legitimate messages
- `test_fusion.py` — weights sum to 1, floor rule, boundary conditions

---

# 🔐 Privacy Guarantees

FinGuard AI is designed so that **no financial data ever leaves the device**.

## What stays local

- All uploaded CSVs
- All pasted messages
- All computed scores and explanations
- All model weights (after the initial one-time download)

## What leaves the device

**Nothing.**

## How this is enforced

- No `openai`, `anthropic`, `google-generativeai`, or hosted-inference SDKs are imported anywhere in the codebase
- Models are loaded from local disk via `transformers` and `llama-cpp-python`
- The Streamlit server binds to `localhost`
- The Privacy Proof panel reports zero cloud calls by construction

## Offline operation

You can disconnect Wi-Fi after the first run and the app will continue working fully. This is a deliberate design property and a core requirement of the Ryze AI Hack.

---

# 🛡️ Safety Policy

FinGuard AI is a **safety and decision-support tool**. It is **not**:

- An investment advisor
- A loan approval or rejection system
- A definitive fraud detector
- A replacement for a bank or financial institution

## Language guidelines

The application consistently uses phrases like:

- "Potential risk detected"
- "Unusual activity"
- "Review recommended"

It avoids phrases like:

- "This is fraud"
- "You should invest"
- "100% accurate"
- "Guaranteed"

## Explanation filter

Every LLM-generated explanation passes through a post-generation filter that rejects outputs containing investment advice, guarantees, or definitive claims.

---

# 🧰 Troubleshooting

## ❌ `streamlit: command not found`

Use:

```bat
python -m streamlit run app.py
```

## ❌ `ModuleNotFoundError: modules.X`

Confirm the virtual environment is active (`(.venv)` in prompt) and you are in the project root directory.

## ❌ `llama-cpp-python` fails to build

On Windows without MSVC build tools, install the prebuilt CPU wheel:

```bat
pip install llama-cpp-python --prefer-binary --extra-index-url https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/basic/cpu --force-reinstall --no-deps --no-cache-dir
```

The app works without `llama-cpp-python` — it falls back to the template engine.

## ❌ `OSError: [WinError -1073741795] Windows Error 0xc000001d`

This is an **illegal instruction** error from an AVX-optimized wheel on a non-AVX CPU. Install the **basic** wheel instead (see previous fix).

## ❌ `AssertionError` on LLM load

Means the installed `llama-cpp-python` version doesn't support the model's architecture. The app gracefully falls back to the template engine. No user action needed.

## ❌ CSV upload fails with "Could not identify required 'date' and/or 'amount' columns"

The CSV doesn't have recognizable date/amount columns. Rename them to `date` and `amount`, or add aliases (see `modules/ingest.py`).

## ❌ Generated explanations are empty

Check `modules/explain.py` for a syntax issue. The template engine has no external dependencies and should always produce output.

## ❌ Slow performance

- Reduce the transaction CSV size (500 rows is enough for a demo)
- Disable the optional LLM path
- Close other applications — the Isolation Forest and DistilBERT both use CPU

---

# 📦 Main Dependencies

| Dependency | Purpose |
|---|---|
| pandas | Data manipulation |
| numpy | Numerics |
| scikit-learn | Isolation Forest |
| pyyaml | Config loading |
| streamlit | Web dashboard |
| transformers | Optional DistilBERT scam embeddings |
| torch | Backend for transformers |
| llama-cpp-python | Optional local LLM inference |
| psutil | Privacy proof RAM monitoring |
| tabulate | Evaluation report formatting |

---

# 🧩 Technology Stack

```text
Frontend
└── Streamlit

Backend
└── Python 3.11+

AI / ML
├── scikit-learn (Isolation Forest)
├── DistilBERT (optional scam embeddings, 66M)
└── Qwen2.5-0.5B (optional explanation, 494M)

Statistics
└── pandas + numpy (behavior analysis)

Privacy
└── 100% local — no cloud APIs

Deployment
├── Local Python
└── Optional Docker
```

---

# 🔄 End-to-End Workflow

```text
                 ┌───────────────────┐
                 │   User Input      │
                 │  (CSV + Messages) │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │   Streamlit UI    │
                 └─────────┬─────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
  Scam Scanner      Anomaly Detector    Behavior
  (rules + opt.      (Isolation        (statistics)
   embeddings)        Forest)
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │  Fusion Engine    │
                 │  (weighted sum)   │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │  AI Explanation   │
                 │  (template | LLM) │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │   Dashboard       │
                 │   (7 sections)    │
                 └───────────────────┘
```

---

# 📈 Future Roadmap

Potential future improvements:

- [ ] Fine-tuned sentence-transformer for the scam module
- [ ] Voice alert for high-risk transactions
- [ ] Bank-statement PDF parser
- [ ] Urdu / Roman-Urdu scam message templates
- [ ] Time-series forecasting for spending trends
- [ ] User-tunable fusion weights via the UI
- [ ] Automatic anomaly labeling from user feedback
- [ ] Visualize anomaly scores on a timeline chart
- [ ] Multi-account transaction merging
- [ ] Exportable PDF risk reports
- [ ] Full offline evaluation suite with locked seeds
- [ ] Cross-platform packaging (PyInstaller)
- [ ] WebAssembly deployment for browser-only mode
- [ ] Extended scam database for Pakistani banks
- [ ] Automated PII redaction before analysis

---

# 🧑‍💻 Development

Run the app directly:

```bat
streamlit run app.py
```

Run the module self-tests (each module has a `python -m modules.X` entry point):

```bat
python -m modules.ingest
python -m modules.anomaly
python -m modules.behavior
python -m modules.scam
python -m modules.fusion
python -m modules.explain
```

Recommended `.gitignore` entries:

```gitignore
.venv/
venv/
__pycache__/
*.pyc
.env
models/**/*.gguf
models/**/*.bin
models/**/*.safetensors
evaluation/results/*.json
evaluation/results/*.csv
.streamlit/secrets.toml
.DS_Store
Thumbs.db
.vscode/
.idea/
```

Never commit model weights or real financial CSVs.

---

# 📜 License

Add an appropriate license before publicly distributing this repository. For a permissive open-source license, consider **MIT** or **Apache-2.0**.

Also review the licenses of third-party models and libraries used by the project (DistilBERT, Qwen2.5, scikit-learn, Streamlit, etc.). A project license does not automatically grant rights to third-party models.

---

# 🙌 Acknowledgements

This project builds on the work of:

- **scikit-learn** — Isolation Forest implementation
- **Hugging Face Transformers** — DistilBERT weights and tokenization
- **Qwen team** — Qwen2.5-0.5B-Instruct GGUF weights
- **llama.cpp** and **llama-cpp-python** — local LLM inference
- **Streamlit** — the dashboard framework

Please review the respective model and library licenses before redistribution.

---

# 🏆 Hackathon Context

Built for the **Ryze AI Hack 2026**.

Compliant with all hackathon constraints:

- ✅ Every AI model is ≤500M parameters
- ✅ All inference runs locally on the user's device
- ✅ Zero cloud AI APIs are called
- ✅ The application remains fully functional offline
- ✅ Meaningful on-device AI is demonstrated (unsupervised ML + optional local LLM)

---

# ⭐ GitHub Repository

If this project is useful for your research, portfolio, or financial-safety exploration:

```text
⭐ Star the repository
🍴 Fork it
🐛 Report reproducible bugs
💡 Suggest improvements
🔧 Submit pull requests
```

---

# 👨‍💻 Author

***Zohaib Sattar***

Built as a privacy-first AI engineering project combining:

```text
Artificial Intelligence
+
Machine Learning
+
Financial Safety
+
Anomaly Detection
+
Natural Language Processing
+
On-Device Inference
+
Python
```

---

<div align="center">

### 🛡️ Your money. Your device. Your privacy.

**FinGuard AI**

</div>
```
