# This project was submitted to the ryze.ai hackathon by Zohaib Sattar.

# 🛡️ FinGuard AI

<div align="center">

### 🔒 Privacy-First Financial Safety • 💳 Transaction Anomaly Detection • 📱 Scam Detection • 📊 Spending Analysis • 🧠 Risk Fusion • 🗣️ Local AI

**Your money. Your device. Your privacy.**

A privacy-first, fully local financial safety agent that analyzes transactions, spending behavior, and financial messages directly on the user's device.

**No cloud AI APIs. No financial data uploads. Works offline.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python\&logoColor=white)](#requirements)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit\&logoColor=white)](#application-features)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikit-learn\&logoColor=white)](#transaction-anomaly-detector)
[![Transformers](https://img.shields.io/badge/NLP-DistilBERT-FFD21E)](#scam-message-scanner)
[![llama.cpp](https://img.shields.io/badge/LLM-llama.cpp-black)](#local-ai-explanation)
[![Offline](https://img.shields.io/badge/Offline-100%25-1a7f37)](#privacy-proof-mode)
[![Cloud AI](https://img.shields.io/badge/Cloud%20AI%20APIs-0-a01010)](#privacy-proof-mode)

</div>

---

## 📌 Overview

**FinGuard AI** is a privacy-first financial safety system designed to run entirely on the user's device.

It analyzes three types of financial signals:

* Financial messages
* Transaction history
* Spending behavior

These signals are processed independently and combined into a transparent **Overall Financial Risk Score (0–100)**.

FinGuard then provides a plain-language explanation of why a risk was detected.

The system is designed for the **Ryze AI Hack 2026**, where AI models must have **500M parameters or fewer** and inference must run locally without cloud AI APIs.

> **Important:** FinGuard AI is a financial safety and decision-support tool. It does not provide investment advice, approve or reject loans, or replace a bank or financial institution. Risk scores indicate potential risk and should be reviewed by the user.

---
## 📸 Dashboard

![FinGuard AI Dashboard](![Uploading Screenshot 2026-10-06 040218.png…]()
)


# ✨ Key Features

| Feature                         | Purpose                                    |
| ------------------------------- | ------------------------------------------ |
| 📱 Scam Message Scanner         | Detects potential financial scam signals   |
| 💳 Transaction Anomaly Detector | Finds unusual transactions                 |
| 📊 Spending Behavior Analyzer   | Identifies spending trends and deviations  |
| 🧠 Risk Fusion Engine           | Combines module-level signals              |
| 🗣️ Local AI Explanation        | Explains detected risks in simple language |
| 🔒 Privacy Proof Mode           | Demonstrates local and offline processing  |

---

# 🏗️ System Architecture

```text
                    USER FINANCIAL DATA
                            │
             ┌──────────────┼──────────────┐
             │              │              │
             ▼              ▼              ▼
      Scam Scanner     Transaction     Spending /
                       Anomaly        Behavior
                       Detector       Analyzer
             │              │              │
             └──────────────┼──────────────┘
                            │
                            ▼
                    Risk Fusion Engine
                            │
                            ▼
                 Overall Risk Score
                     LOW / MEDIUM / HIGH
                            │
                            ▼
                  Local AI Explanation
                            │
                            ▼
                    Streamlit Dashboard
```

---

#  Application Features

## 1. 📱 Scam Message Scanner

The Scam Scanner analyzes SMS or pasted financial messages and identifies potential scam signals.

### Supported categories

* OTP or PIN requests
* Credential-sharing requests
* KYC-related scams
* Account, SIM, or card blocking threats
* Fake prizes and lottery messages
* Fake loan offers
* Suspicious URLs
* Phishing links
* Cashback/refund scams
* Bank impersonation
* Suspicious payment requests
* Urgency or threat language
* Guaranteed-return or crypto-related bait

### Detection approach

The primary scanner uses a lightweight local rule engine.

Each detected pattern contributes a weighted risk signal.

| Signal                         | Weight |
| ------------------------------ | -----: |
| OTP / PIN request              |   1.00 |
| Credential sharing request     |   1.00 |
| KYC blocking/re-KYC urgency    |   0.85 |
| Account/SIM/card blocking      |   0.80 |
| Prize/lottery bait             |   0.85 |
| Instant/pre-approved loan bait |   0.75 |
| IP-address URL                 |   0.75 |
| Suspicious/phishing link       |   0.70 |
| Cashback/refund bait           |   0.65 |
| Bank impersonation             |   0.60 |
| Payment request                |   0.55 |
| Urgency/threat language        |   0.55 |
| Crypto/guaranteed-return bait  |   0.40 |

The rule score is converted to a 0–100 risk score using:

```text
score = (1 - exp(-total_weight)) × 100
```

The scanner returns:

```text
LEGITIMATE
SUSPICIOUS
SCAM
```

### Optional NLP signal

FinGuard also supports an optional local **DistilBERT-base-uncased** component with approximately **66M parameters**.

This component is optional and disabled by default.

The rule engine remains the primary deterministic signal because it is lightweight, transparent, and reproducible.

---

# 2. 💳 Transaction Anomaly Detector

The Transaction Anomaly Detector analyzes uploaded transaction history and identifies transactions that significantly differ from normal behavior.

### Features

The detector can use:

* `log1p(amount)`
* Transaction hour
* Day of week
* Merchant frequency
* New merchant indicator
* Rolling amount ratio
* Daily transaction count

### Algorithm

FinGuard uses **Isolation Forest** for transaction-level anomaly detection.

Default configuration:

```text
Trees:        200
Contamination: 0.05
```

Isolation Forest is suitable because it:

* Does not require fraud labels
* Works locally
* Handles numerical transaction features
* Is lightweight
* Produces transaction-level anomaly scores

For very small datasets, the system falls back to a statistical z-score approach.

### Output

Each transaction receives:

```text
anomaly_raw
anomaly_score
is_anomaly
```

Example:

```text
Transaction:
Rs 85,000 → Unknown Merchant

Anomaly Score:
94 / 100

Status:
⚠️ Potentially unusual
```

---

# 3. 📊 Spending Behavior Analyzer

The Spending Behavior Analyzer uses statistics rather than an LLM.

It is designed to identify changes in the user's normal spending behavior.

### Analysis

The module calculates:

* Monthly spending totals
* Transaction counts
* Average transaction amount
* Category-level spending
* Merchant-level spending
* Recurring expenses
* Current-month deviation
* Spending spikes
* Historical baseline

### Example

```text
Historical monthly spending:
Rs 70,000

Current month:
Rs 115,000

Deviation:
+64.3%

Risk:
⚠️ Significant spending increase
```

### Recurring expense detection

A recurring expense is identified when:

* The same merchant appears at least 3 times
* Amount remains within approximately ±10% of the median
* Transaction intervals resemble weekly or monthly cycles

Recurring expenses are reported as **informational insights** rather than automatic risk signals.

---

# 4. 🧠 Risk Fusion Engine

The Risk Fusion Engine combines the outputs of the three main modules:

```text
Scam Risk
Transaction Risk
Behavior Risk
        │
        ▼
   Risk Fusion
        │
        ▼
Overall Risk Score
```

### Default weights

```text
Scam         40%
Transaction  35%
Behavior     25%
```

The overall score is calculated as:

```text
Overall Risk =
    (Scam Risk × 0.40)
  + (Transaction Risk × 0.35)
  + (Behavior Risk × 0.25)
```

The weights are configurable.

### Risk levels

|  Score | Risk Level |
| -----: | ---------- |
|   0–39 | 🟢 LOW     |
|  40–69 | 🟠 MEDIUM  |
| 70–100 | 🔴 HIGH    |

### Risk floor

A single strong signal should not disappear inside the weighted average.

Therefore:

```text
If any module ≥ 70:
    Overall risk cannot be LOW

If any module ≥ 85:
    Overall risk cannot be LOW
```

This ensures that a severe individual signal remains visible to the user.

---

# 5. 🗣️ Local AI Explanation

FinGuard provides a plain-language explanation of the detected risk.

There are two explanation backends.

## Default: Deterministic Template Engine

The template engine:

* Requires no model
* Has near-zero latency
* Works completely offline
* Produces reproducible explanations
* Works on low-end hardware

Example:

> Potential high financial risk detected with a score of 81/100. This is primarily due to multiple high-risk message signals and transactions that differ significantly from your usual activity. Review the flagged items before making a payment.

## Optional: Local LLM

FinGuard can optionally use:

**Qwen2.5-0.5B-Instruct**

```text
Parameters: approximately 494M
Format: GGUF Q4_K_M
Approximate size: 407 MB
Runtime: llama.cpp
```

The model runs locally and does not require a cloud API.

### Safety filtering

Generated explanations are checked before being displayed.

The system prevents or rewrites language that:

* Gives investment advice
* Guarantees financial outcomes
* Makes definitive fraud claims
* Approves or rejects loans
* Uses unsupported certainty

Examples:

```text
"This is definitely fraud."
        ↓
"This message shows strong scam signals."

"You should invest in..."
        ↓
Blocked / replaced
```

If the local LLM is unavailable, FinGuard automatically uses the deterministic template engine.

---

# 6. 🔒 Privacy Proof Mode

Privacy is a core feature rather than just a claim.

FinGuard provides a dedicated Privacy Proof panel showing:

| Metric                  | Expected Value  |
| ----------------------- | --------------- |
| Cloud AI API calls      | **0**           |
| Financial data uploaded | **0 bytes**     |
| Processing location     | **This device** |
| Cloud inference         | **Disabled**    |
| Offline operation       | **Supported**   |

The application is designed to continue operating after the required model files have been downloaded and the device is disconnected from the internet.

---

# 🔐 Privacy Architecture

## Data that stays local

The following information remains on the user's device:

* Uploaded CSV files
* Financial messages
* Transaction records
* Calculated risk scores
* Spending statistics
* Generated explanations
* Local model files

## Cloud services

FinGuard does not require:

* OpenAI API
* Anthropic API
* Google Gemini API
* Hosted Hugging Face inference
* Other cloud inference services

The core application performs processing locally.

---

# 🧱 Project Structure

```text
finguard-ai/
│
├── README.md
├── requirements.txt
├── requirements-lock.txt
├── .gitignore
├── app.py
├── privacy_proof.py
│
├── config/
│   └── fusion_weights.yaml
│
├── modules/
│   ├── __init__.py
│   ├── ingest.py
│   ├── anomaly.py
│   ├── behavior.py
│   ├── scam.py
│   ├── fusion.py
│   └── explain.py
│
├── models/
│   ├── scam_detector/
│   ├── explainer/
│   └── README.md
│
├── data/
│   ├── sample_transactions.csv
│   └── sample_messages.csv
│
├── evaluation/
│   ├── evaluate.py
│   └── results/
│       └── baseline.md
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

* Python 3.11+
* pip
* Windows, Linux, or macOS
* Internet connection for initial model downloads, if optional models are enabled

After required model files are available locally, the application can operate offline.

## Hardware

| Component | Minimum      | Recommended         |
| --------- | ------------ | ------------------- |
| CPU       | Dual-core    | Quad-core with AVX2 |
| RAM       | 4 GB         | 8 GB+               |
| Disk      | ~1 GB        | 2 GB+               |
| GPU       | Not required | Not required        |

### Optional LLM

The Qwen/llama.cpp backend may require CPU instruction-set support depending on the selected wheel.

On unsupported hardware, FinGuard automatically uses the deterministic explanation engine.

---

# 🤖 On-Device Model Budget

All AI models used by FinGuard must remain within the Ryze AI Hack's **500M parameter limit**.

| Component            | Technology       | Parameters |
| -------------------- | ---------------- | ---------: |
| Scam detection       | Rule engine      |          0 |
| Optional scam NLP    | DistilBERT       |       ~66M |
| Transaction analysis | Isolation Forest |        N/A |
| Behavior analysis    | Statistics       |        N/A |
| Risk fusion          | Arithmetic       |        N/A |
| Optional explanation | Qwen2.5-0.5B     |      ~494M |

> Traditional algorithms such as Isolation Forest and statistical analysis are not neural language models and do not have a neural parameter count.

---

# 🪟 Installation — Windows

## 1. Clone the repository

```bat
git clone https://github.com/YOUR_USERNAME/finguard-ai.git
cd finguard-ai
```

Replace `YOUR_USERNAME` with your GitHub username.

## 2. Create a virtual environment

```bat
python -m venv .venv
.venv\Scripts\activate
```

Verify that the terminal shows:

```text
(.venv)
```

## 3. Install dependencies

```bat
pip install -r requirements.txt
```

For CPU-only PyTorch installations:

```bat
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 4. Start FinGuard

```bat
streamlit run app.py
```

Open:

```text
http://localhost:8501
```

---

# 🐧 Linux / macOS

```bash
git clone https://github.com/YOUR_USERNAME/finguard-ai.git
cd finguard-ai

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

streamlit run app.py
```

---

# 🧪 Using FinGuard

## Step 1 — Load transactions

From the sidebar:

* Upload a bank CSV
* Or select **Load sample transactions**

## Step 2 — Add messages

Paste one or more financial messages into the Scam Scanner.

Example:

```text
Dear customer, your account has been blocked.
Share your OTP now.

You have won Rs 500000!
Click the link to claim your prize.

Your KYC is pending.
Update your account immediately.
```

## Step 3 — Run analysis

Click:

```text
🔍 Scan Messages
```

FinGuard processes the available signals locally.

## Step 4 — Review dashboard

The dashboard presents:

1. Overall Financial Risk
2. Scam Scanner
3. Transaction Analysis
4. Spending Behavior
5. Risk Alerts
6. AI Explanation
7. Privacy Proof

---

# 📊 Dashboard

## Overall Financial Risk

Displays:

```text
Overall Risk
76 / 100

🔴 HIGH
```

An expandable section shows how each module contributed to the final score.

## Scam Scanner

Shows:

* Message
* Classification
* Risk score
* Detected signals

Example:

| Label | Score | Signals                                |
| ----- | ----: | -------------------------------------- |
| SCAM  |    93 | OTP, credential request, account block |
| SCAM  |    78 | Prize bait, phishing link              |
| SCAM  |    90 | KYC block, suspicious URL              |

## Transaction Analysis

Shows the highest-risk transactions including:

* Date
* Amount
* Merchant
* Anomaly score
* Flag status

## Spending Behavior

Shows:

* Monthly totals
* Spending deviation
* Category breakdown
* Recurring expenses
* Spending spikes

## Risk Alerts

Provides concise alerts for important findings.

## AI Explanation

Displays:

* Explanation
* Backend used
* Generation latency
* Whether fallback was used

## Privacy Proof

Displays local-processing indicators such as:

```text
Cloud AI APIs: 0
Financial data uploaded: 0 bytes
Processing: This device
```

---

# 📈 Evaluation

Run:

```bat
python evaluation/evaluate.py
```

The evaluation pipeline should measure:

### Scam module

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

### Transaction module

* Anomaly detection rate
* False-positive rate
* Detection latency

### System

* Module latency
* RAM usage
* Model size
* Parameter count
* Offline functionality

Results are saved under:

```text
evaluation/results/
```

> Evaluation results should be reported honestly. Small hand-crafted test sets should be treated as sanity checks, not as statistically representative benchmarks.

---

# 🧪 Testing

Install pytest:

```bat
pip install pytest
```

Run:

```bat
pytest tests/ -v
```

### Test coverage

```text
test_ingest.py
├── Date parsing
├── Amount parsing
├── Missing columns
└── Empty CSV handling

test_anomaly.py
├── Synthetic anomalies
├── Deterministic output
└── Latency

test_behavior.py
├── Recurring expenses
├── Monthly deviation
└── Spending spikes

test_scam.py
├── OTP signals
├── KYC signals
├── Prize signals
├── Phishing signals
└── Legitimate messages

test_fusion.py
├── Weight validation
├── Risk boundaries
└── Floor-rule behavior
```

---

# 🔄 End-to-End Workflow

```text
                    USER INPUT
                 CSV + Messages
                        │
                        ▼
                ┌──────────────┐
                │ Streamlit UI │
                └──────┬───────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
   Scam Scanner   Anomaly Detector  Behavior
   Rules / NLP    Isolation Forest  Statistics
        │              │              │
        └──────────────┼──────────────┘
                       │
                       ▼
               Risk Fusion Engine
                       │
                       ▼
              Overall Risk Score
                       │
                       ▼
              Local Explanation
                       │
                       ▼
                 Dashboard
```

---

# 🛡️ Safety Policy

FinGuard AI is a financial safety and decision-support application.

It is **not**:

* An investment advisor
* A loan approval system
* A definitive fraud detector
* A replacement for a bank
* A financial guarantee system

### Preferred language

Use:

* "Potential risk detected"
* "Unusual activity"
* "Review recommended"
* "Strong scam signals detected"

Avoid:

* "This is definitely fraud"
* "You should invest"
* "Guaranteed"
* "100% accurate"
* "Your loan should be approved"

---

# 🧰 Troubleshooting

## `streamlit: command not found`

Use:

```bat
python -m streamlit run app.py
```

## `ModuleNotFoundError: modules.X`

Make sure:

1. The virtual environment is activated.
2. You are running commands from the project root.
3. `modules/__init__.py` exists.

## CSV column detection fails

FinGuard expects recognizable aliases for:

```text
date
amount
merchant
category
```

If automatic detection fails, rename the columns accordingly.

## `llama-cpp-python` installation fails

The optional LLM backend is not required for the core application.

FinGuard can use the deterministic explanation engine instead.

## LLM crashes with illegal instruction

This usually indicates an incompatible CPU/wheel combination.

Disable the optional LLM backend and use the template explanation engine.

---

# 📦 Dependencies

| Package          | Purpose                    |
| ---------------- | -------------------------- |
| pandas           | Data processing            |
| numpy            | Numerical computation      |
| scikit-learn     | Isolation Forest           |
| pyyaml           | Configuration              |
| streamlit        | Dashboard                  |
| transformers     | Optional DistilBERT        |
| torch            | Transformer backend        |
| llama-cpp-python | Optional local LLM         |
| psutil           | System/resource monitoring |
| tabulate         | Evaluation reports         |

---

# 🧩 Technology Stack

```text
Frontend
└── Streamlit

Backend
└── Python 3.11+

Machine Learning
├── scikit-learn
│   └── Isolation Forest
│
└── Optional DistilBERT
    └── Financial message NLP

Behavior Analysis
├── pandas
└── numpy

Local Explanation
└── Optional Qwen2.5-0.5B
    └── llama.cpp

Privacy
└── Local processing
    └── No cloud AI APIs
```

---

# 🔐 Privacy by Design

FinGuard is designed around a simple principle:

> **Financial data should stay where it belongs — on the user's device.**

The application processes locally:

* Financial CSVs
* Financial messages
* Transaction features
* Risk calculations
* Spending statistics
* Explanations

No financial information is intentionally sent to a cloud AI service.

The application can also be demonstrated with the internet disabled after the required local model files are available.

---

# 🏆 Ryze AI Hack 2026

FinGuard AI was built for the **Ryze AI Hack 2026**.

### Design goals

* ✅ Local AI
* ✅ Models ≤500M parameters
* ✅ No cloud AI inference
* ✅ Offline operation
* ✅ Privacy-first financial analysis
* ✅ Meaningful on-device AI
* ✅ Lightweight consumer-hardware deployment
* ✅ Explainable risk scoring

The project deliberately combines:

```text
Traditional ML
      +
Statistical Analysis
      +
NLP
      +
Local LLM
      +
Transparent Risk Fusion
```

rather than forcing every problem through an LLM.

---

# 📈 Future Roadmap

Potential improvements:

* [ ] Fine-tuned financial scam classifier
* [ ] Urdu and Roman-Urdu scam detection
* [ ] Pakistani bank and wallet scam dataset
* [ ] Bank-statement PDF extraction
* [ ] Spending time-series forecasting
* [ ] Timeline visualization of anomalies
* [ ] Multi-account analysis
* [ ] User feedback for anomaly labeling
* [ ] User-configurable risk weights
* [ ] Exportable risk reports
* [ ] Automatic PII detection and redaction
* [ ] Voice alerts for high-risk events
* [ ] Cross-platform packaging with PyInstaller

---

# 🧑‍💻 Development

Run the application:

```bat
streamlit run app.py
```

Run individual module checks:

```bat
python -m modules.ingest
python -m modules.anomaly
python -m modules.behavior
python -m modules.scam
python -m modules.fusion
python -m modules.explain
```

Run the full test suite:

```bat
pytest tests/ -v
```

Run evaluation:

```bat
python evaluation/evaluate.py
```

---

# 🚫 Git & Data Safety

Never commit:

* Real bank statements
* Real financial transactions
* Personal SMS messages
* API keys
* `.env` files
* Model credentials
* Private user information

Recommended `.gitignore`:

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

.vscode/
.idea/
.DS_Store
Thumbs.db
```

> Model weights may be distributed separately when required by the project and their licenses permit redistribution.

---

# 📜 License

Choose an appropriate project license before public distribution, such as:

* MIT
* Apache-2.0

Also review the licenses of all third-party models, datasets, and libraries used by the project.

A project license does not automatically grant redistribution rights for third-party model weights or datasets.

---

# 🙌 Acknowledgements

FinGuard AI builds on open-source technologies including:

* **scikit-learn** — machine-learning algorithms
* **Hugging Face Transformers** — transformer models and tokenization
* **Qwen** — local language model
* **llama.cpp / llama-cpp-python** — local LLM inference
* **Streamlit** — interactive dashboard
* **pandas / NumPy** — data processing and numerical analysis

Please review the respective licenses before redistribution.

---

# 👨‍💻 Author

## Zohaib Sattar

FinGuard AI combines:

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
On-Device AI
        +
Privacy Engineering
```

---

<div align="center">

## 🛡️ FinGuard AI

### Your money. Your device. Your privacy.

**Built for the Ryze AI Hack 2026**

</div>
