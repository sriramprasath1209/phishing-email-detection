# PhishGuard — Phishing Email Detection Using DFA and SLM

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Framework: Flask](https://img.shields.io/badge/framework-Flask-black.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests: 27 Passing](https://img.shields.io/badge/tests-27%20passing-brightgreen.svg)](tests/)

**PhishGuard** is a full-stack cybersecurity web application designed to detect and triage phishing emails. It combines a formal **Deterministic Finite Automaton (DFA)** state machine for exact pattern and URL structural analysis, a custom-trained **TF-IDF + Logistic Regression Machine Learning Pipeline**, and a modular, prompt-injection-defended interface for future **Small Language Model (SLM)** integration.

Designed with a modern, dark-themed cybersecurity dashboard, PhishGuard provides actionable risk scoring, component breakdown, salient feature explanations, and transparent decision rationales.

---

## Table of Contents

1. [Key Features](#key-features)
2. [Architecture Overview](#architecture-overview)
3. [Technology Stack](#technology-stack)
4. [Project Structure](#project-structure)
5. [Prerequisites & macOS Setup](#prerequisites--macos-setup)
6. [Installation & Quickstart](#installation--quickstart)
7. [Dataset Preparation & Custom Model Training](#dataset-preparation--custom-model-training)
8. [Running Automated Tests](#running-automated-tests)
9. [REST API Documentation & Examples](#rest-api-documentation--examples)
10. [Small Language Model (SLM) Integration Guide](#small-language-model-slm-integration-guide)
11. [Limitations & Adversarial Considerations](#limitations--adversarial-considerations)
12. [Git & GitHub Setup](#git--github-setup)

---

## 1. Key Features

- **Multi-Modal Detection Engine**:
  - **Deterministic Finite Automaton (DFA)**: Scans email text via an $O(N)$ single-pass prefix automaton for multi-token phishing phrases (e.g., *"verify your account"*, *"confirm your password"*, *"wire transfer request"*).
  - **URL Structural Scanner DFA**: Inspects raw IP addresses in URL hostnames (e.g., `http://192.168.1.1/login`), `@` userinfo spoofing (e.g., `http://chase.com@phish.xyz`), non-standard HTTP ports, hex/percent encoding, and high-abuse top-level domains (`.xyz`, `.top`, `.click`).
  - **Interpretable Machine Learning**: Custom TF-IDF vectorizer + Logistic Regression pipeline fitted with strict anti-leakage boundaries and salient feature attribution.
  - **Modular SLM Interface**: Prompt-injection-isolated architectural interface for qualitative intent and coercion auditing using local Small Language Models (e.g. Llama-3.2-1B, Phi-3.5-mini).
- **Transparent Risk Scoring**:
  - Generates a calibrated Risk Score ($0 - 100$) and four risk classifications: `LIKELY_LEGITIMATE` ($<30$), `SUSPICIOUS` ($30-65$), `PHISHING` ($>65$), or `INCONCLUSIVE`.
  - Enforces a critical risk floor when high-severity DFA indicators (e.g., raw IP hosts or password harvest prompts) are triggered.
- **Reduced Capability Mode**:
  - If the machine learning model is uninitialized or missing, the system operates in a clearly labeled DFA-only mode.
  - Adheres to sound security principles: emails without detected patterns are labeled `INCONCLUSIVE` rather than blindly declared "safe".
- **Enterprise-Grade Cybersecurity Dashboard**:
  - Dark navy palette (`#070b14`), blue/violet accent lighting, and responsive typography (Inter & JetBrains Mono).
  - One-click presets for legitimate meeting notes, banking alerts, courier delivery scams, and executive wire fraud.
  - Strict client-side security: uses `textContent` rendering throughout to prevent Cross-Site Scripting (XSS); defangs links so they cannot be clicked accidentally.

---

## 2. Architecture Overview

```
                          ┌──────────────────────────┐
                          │  Email Input (User / UI) │
                          └─────────────┬────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │   Input Validation & Size   │
                         │   Limits (Flask REST API)   │
                         └──────────────┬──────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              ▼                         ▼                         ▼
    ┌───────────────────┐     ┌───────────────────┐     ┌───────────────────┐
    │   DFA Detection   │     │  Machine Learning │     │    Modular SLM    │
    │      Module       │     │    Classifier     │     │     Interface     │
    │ (Tokens & URLs)   │     │ (TF-IDF + LogReg) │     │ (Prompt-Isolated) │
    └─────────┬─────────┘     └─────────┬─────────┘     └─────────┬─────────┘
              │                         │                         │
              └─────────────────────────┼─────────────────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │   Combined Detector Engine  │
                         │   • Risk Scoring (0 - 100)  │
                         │   • Mode Auto-Switching     │
                         │   • Transparent Rationale   │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                         ┌─────────────────────────────┐
                         │  Cybersecurity Dashboard UI │
                         │  (Vanilla JS / Safe DOM)    │
                         └─────────────────────────────┘
```

For full mathematical definitions, state transition graphs, and tables, see [`docs/architecture.md`](file:///Users/sriramprasath/Documents/Phishing%20Email%20Detection/docs/architecture.md).

---

## 3. Technology Stack

- **Frontend**: HTML5, Vanilla CSS3 (Custom Design System), Vanilla JavaScript (ES6+), Fetch API.
- **Backend**: Python 3.10+, Flask, REST API.
- **Machine Learning**: Scikit-learn, TF-IDF Vectorization, Logistic Regression, Pandas, NumPy, Joblib.
- **Automated Testing**: Pytest, Flask Test Client.
- **Operating Environment**: macOS, VS Code, Git.

---

## 4. Project Structure

```
Phishing Email Detection/
├── app.py                      # Flask REST API server and routing
├── requirements.txt            # Python package dependencies
├── README.md                   # Complete documentation and setup manual
├── .gitignore                  # Git exclusions for venv, cache, and artifacts
├── config.py                   # Central security, threshold, and scoring configs
├── dfa_detector.py             # Formal DFA engine (PhraseDFA & URLStructureDFA)
├── model_detector.py           # ML inference engine and salient feature extractor
├── train_model.py              # ML training pipeline with anti-leakage splitting
├── combined_detector.py        # Multi-modal fusion and risk score calculator
├── slm_detector.py             # Modular Small Language Model interface
├── templates/
│   └── index.html              # Accessible cybersecurity dashboard UI
├── static/
│   ├── style.css               # Dark navy design system
│   └── script.js               # Client-side validation, presets, and safe DOM
├── data/
│   ├── README.md               # Dataset sourcing and format guide
│   └── sample_dataset.csv      # Curated dataset for training validation
├── models/
│   ├── .gitkeep                # Tracks models directory in Git
│   └── phishing_model.joblib   # Trained scikit-learn model artifact
├── tests/
│   ├── __init__.py             # Test package initialization
│   ├── test_api.py             # API input validation and endpoint tests
│   ├── test_dfa.py             # Formal DFA transition and pattern tests
│   └── test_model.py           # Model loader, training, and inference tests
└── docs/
    └── architecture.md         # Comprehensive system specification
```

---

## 5. Prerequisites & macOS Setup

- **Operating System**: macOS Sonoma or later (Apple Silicon / Intel).
- **Python**: Python 3.10, 3.11, 3.12, or 3.14.
- **Terminal**: `zsh` or `bash`.

Verify your Python installation in Terminal:
```bash
python3 --version
```

---

## 6. Installation & Quickstart

### Step 1: Clone or Navigate to the Workspace
```bash
cd "/Users/sriramprasath/Documents/Phishing Email Detection"
```

### Step 2: Create and Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Train the Custom Classifier
Train the initial baseline classifier on the curated dataset:
```bash
python train_model.py
```
*This evaluates accuracy, precision, recall, and confusion matrix, saving the pipeline to `models/phishing_model.joblib`.*

### Step 5: Start the Flask Application
```bash
python app.py
```

### Step 6: Access the Dashboard
Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 7. Dataset Preparation & Custom Model Training

### Dataset Format
PhishGuard expects a CSV containing `text` and `label` columns:
- `text`: Email subject and body content.
- `label`: `0` for Legitimate (Ham), `1` for Phishing.

```csv
text,label
"Team sync meeting scheduled for Thursday at 2 PM to review quarterly OKRs.",0
"Your account has been suspended due to suspicious activity. Verify credentials at http://192.168.1.1/auth now.",1
```

### Training Command Options
To train on your own external dataset (e.g. Enron, Nazario corpus, or Kaggle CSVs):
```bash
python train_model.py --data data/your_custom_dataset.csv --output models/phishing_model.joblib --test-size 0.25 --seed 42
```

### Training Pipeline Protections
1. **Deduplication**: Automatically drops identical emails before splitting to eliminate train/test contamination.
2. **Stratification**: Guarantees identical legitimate-to-phishing class distributions across training and test sets.
3. **Pipeline Fitting**: TF-IDF vectorizer fits strictly on the training partition, ensuring zero feature leakage.

---

## 8. Running Automated Tests

PhishGuard includes 27 comprehensive unit and integration tests covering:
- DFA multi-token phrase transitions and case-insensitivity.
- URL character-level DFA checks (IPv4 hosts, userinfo `@` spoofing, non-standard ports, suspicious TLDs).
- Custom ML training pipeline execution and persistence.
- Salient feature extraction and decision threshold mechanics.
- Combined detector in full hybrid mode and reduced capability mode.
- REST API validation: valid requests, empty input (400), malformed JSON (400), invalid types (400), oversized payloads (413), method not allowed (405).

Execute the test suite:
```bash
pytest tests/ -v
```

Expected output:
```
tests/test_api.py::TestAPIEndpoints::test_get_index_page PASSED
tests/test_api.py::TestAPIEndpoints::test_health_check_endpoint PASSED
tests/test_api.py::TestAPIEndpoints::test_analyze_valid_payload PASSED
...
tests/test_dfa.py::TestPhraseDFA::test_exact_phrase_match PASSED
tests/test_dfa.py::TestURLStructureDFA::test_detect_ipv4_host PASSED
...
tests/test_model.py::TestModelTrainingAndInference::test_trained_model_inference_phishing PASSED
============================== 27 passed in 1.01s ==============================
```

---

## 9. REST API Documentation & Examples

### `GET /api/health`
Returns backend health and component availability.

**Example Request:**
```bash
curl -s http://127.0.0.1:5000/api/health
```

**Example Response:**
```json
{
  "status": "healthy",
  "service": "PhishGuard",
  "version": "1.0.0",
  "components": {
    "dfa_engine": {
      "registered_phrases": 74,
      "status": "active"
    },
    "machine_learning_model": {
      "is_loaded": true,
      "model_path": ".../models/phishing_model.joblib",
      "decision_threshold": 0.5,
      "status_message": "Loaded and operational."
    },
    "slm_interface": {
      "enabled": false,
      "is_available": false,
      "model_name": "meta-llama/Llama-3.2-1B-Instruct"
    }
  }
}
```

---

### `POST /api/analyze`
Analyzes email subject and body for phishing indicators.

**Headers**:
- `Content-Type: application/json`

**Payload Schema**:
```json
{
  "subject": "string (required, max 500 chars)",
  "body": "string (required, max 50000 chars)",
  "sender": "string (optional, max 320 chars)"
}
```

**Example Request (Phishing):**
```bash
curl -s -X POST http://127.0.0.1:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "subject": "URGENT: Your account has been suspended",
    "body": "Confirm your password and verify your account at http://192.168.1.50/verify immediately.",
    "sender": "service-alert@chase-support.xyz"
  }'
```

**Example Response (Excerpt):**
```json
{
  "final_classification": "PHISHING",
  "risk_level": "HIGH",
  "risk_score": 75.0,
  "operating_mode": "HYBRID_DFA_AND_ML",
  "is_reduced_capability": false,
  "dfa": {
    "dfa_risk_score": 95.0,
    "indicators_count": 3,
    "detected_phrases": [
      {
        "category": "Credential Harvesting",
        "pattern": "confirm your password",
        "severity": "CRITICAL"
      },
      {
        "category": "Credential Harvesting",
        "pattern": "verify your account",
        "severity": "HIGH"
      }
    ],
    "url_findings": [
      {
        "url": "http://192.168.1.50/verify",
        "host": "192.168.1.50",
        "anomalies": [
          {
            "type": "IP_ADDRESS_HOST",
            "severity": "CRITICAL",
            "explanation": "Host is a raw numerical IP address (192.168.1.50) rather than a registered domain name."
          }
        ]
      }
    ]
  },
  "machine_learning": {
    "prediction": "PHISHING",
    "phishing_probability": 0.7327,
    "decision_threshold": 0.5,
    "contributing_features": [
      {"term": "account", "direction": "Phishing", "weight": 0.0835},
      {"term": "verify", "direction": "Phishing", "weight": 0.0835}
    ]
  },
  "reasons": [
    "URL Anomaly [IP_ADDRESS_HOST]: Host is a raw numerical IP address...",
    "DFA Pattern [Credential Harvesting]: 'confirm your password'...",
    "ML Classifier estimated phishing probability of 73.3%...",
    "Risk floor elevated to 75+ due to critical DFA security violations."
  ],
  "disclaimer": "Disclaimer: PhishGuard analysis results are heuristic and statistical estimates intended for educational and triage assistance. Automated classifications are not guaranteed to be 100% correct; always verify sensitive requests through out-of-band channels."
}
```

---

## 10. Small Language Model (SLM) Integration Guide

PhishGuard provides a modular interface [`slm_detector.py`](file:///Users/sriramprasath/Documents/Phishing%20Email%20Detection/slm_detector.py) ready for integrating small on-premise language models (such as Llama-3.2-1B, Llama-3.2-3B, Phi-3.5-mini, or Gemma-2-2B).

### Why an SLM?
While TF-IDF detects vocabulary frequencies and DFAs match explicit phrases, Small Language Models excel at understanding:
1. Contextual urgency and subtle psychological coercion tactics.
2. Inconsistencies between sender claims and company practices.
3. Conversational spear-phishing (e.g., CEO wire fraud) devoid of typical spam keywords.

### Enabling the SLM Interface
1. Launch an inference server locally using [Ollama](https://ollama.ai) or [vLLM](https://github.com/vllm-project/vllm):
   ```bash
   ollama run llama3.2:1b
   ```
2. Enable PhishGuard SLM environment variables:
   ```bash
   export PHISHGUARD_SLM_ENABLED=true
   export PHISHGUARD_SLM_MODEL="llama3.2:1b"
   export PHISHGUARD_SLM_ENDPOINT="http://localhost:11434/api/generate"
   python app.py
   ```
3. **Prompt Injection Defenses**: The prompt generator encloses all email text within `<UNTRUSTED_EMAIL_CONTENT>` tags and applies explicit instruction hierarchy so adversarial prompts inside the email cannot hijack the evaluator.

---

## 11. Limitations & Adversarial Considerations

- **Statistical Limitations**: TF-IDF models rely on bag-of-words / n-gram frequencies; sophisticated zero-day spear phishing that uses completely novel phrasing may produce lower ML probabilities.
- **DFA Boundary Limitations**: The DFA detects known configured phrases; intentional misspellings (e.g. *"v3rify your acc0unt"*) can evade exact string state transitions unless preprocessed with homoglyph normalization.
- **False Positives**: Legitimate internal IT security broadcasts reminding employees to rotate passwords contain words similar to credential harvesting campaigns. PhishGuard surfaces salient feature tags and decision rationales to help human analysts verify these edge cases.
- **Educational Disclaimer**: PhishGuard is a heuristic and statistical triage aid. Automated tools cannot replace human scrutiny and out-of-band verification.

---

## 12. Git & GitHub Setup

The project includes a `.gitignore` configured to keep virtual environments (`.venv`), binary model caches (`*.joblib`), and temporary files untracked.

To publish to a new GitHub repository:

```bash
git init
git add .
git commit -m "feat: initial commit of PhishGuard DFA & ML phishing detection platform"
git branch -M main
git remote add origin https://github.com/<your-username>/phishguard.git
git push -u origin main
```
