# PhishGuard — Phishing Email Detection System (DFA & Machine Learning)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Framework: Flask](https://img.shields.io/badge/framework-Flask-black.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests: 40 Passing](https://img.shields.io/badge/tests-40%20passing-brightgreen.svg)](tests/)

**PhishGuard** is an open-source, full-stack cybersecurity web application designed for collegiate and academic research demonstrations. It combines a formal **Deterministic Finite Automaton (DFA)** state machine for exact pattern and URL structural analysis, an integrated **TF-IDF + Logistic Regression Machine Learning Pipeline** trained on ~82,077 emails, and a prompt-injection-defended modular interface for future **Small Language Model (SLM)** integration.

---

## Table of Contents

1. [Key Features](#key-features)
2. [Architecture Overview](#architecture-overview)
3. [Trained Model & Dataset Verification](#trained-model--dataset-verification)
4. [Technology Stack](#technology-stack)
5. [Project Directory Structure](#project-directory-structure)
6. [Prerequisites & macOS Setup](#prerequisites--macos-setup)
7. [Installation & Launch Instructions](#installation--launch-instructions)
8. [Automated Test Suite](#automated-test-suite)
9. [REST API Documentation & Endpoints](#rest-api-documentation--endpoints)
10. [DFA State Machine Specification](#dfa-state-machine-specification)
11. [SLM Clarification & Future Integration](#slm-clarification--future-integration)
12. [Limitations, Adversarial Threats & Privacy](#limitations-adversarial-threats--privacy)
13. [Git & GitHub Setup](#git--github-setup)

---

## 1. Key Features

- **Multi-Modal Detection Engine**:
  - **Deterministic Finite Automaton (DFA)**: Implements an explicit $O(N)$ single-pass prefix automaton for high-confidence multi-token threat phrases (e.g., *"verify your account"*, *"confirm your password"*, *"account has been suspended"*, *"wire transfer request"*).
  - **URL Structural Scanner DFA**: Character-by-character state machine detecting raw IPv4 host addresses (`http://192.168.1.1/login`), `@` userinfo brand spoofing (`http://chase.com@phish.xyz`), non-standard HTTP ports, hex/percent obfuscation, and suspicious top-level domains (`.xyz`, `.top`, `.click`).
  - **Trained Machine Learning Model**: Custom Scikit-learn Pipeline (`TfidfVectorizer` + `LogisticRegression`) trained on ~82,077 clean email records with empirical label verification.
  - **Explainability**: Extracts salient words and bigrams influencing model decisions with directional weights (`Phishing` vs. `Legitimate`).
- **Conservative Decision Policy & Disagreement Handling**:
  - Resolves conflicts between rule-based DFA indicators and statistical ML probabilities (e.g. if an email contains explicit password harvest prompts or raw IP hosts, a critical risk floor of $\ge 75$ is enforced regardless of lower statistical confidence).
- **Reduced Capability Mode**:
  - If the trained model file is absent, the system transparently shifts to DFA-only mode.
  - Avoids false senses of security: emails without detected patterns are labeled **`Analysis Unavailable`** (`INCONCLUSIVE`) rather than "safe" when ML validation is inactive.
- **Cybersecurity Dashboard**:
  - Dark navy theme (`#070b14`), blue/violet accent lighting, and responsive typography (Inter & JetBrains Mono).
  - Pre-configured test lures for instant testing.
  - Safe client-side rendering via `textContent` (zero `innerHTML` injection) and defanged URLs to prevent accidental link execution.

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
                         │   • Disagreement Resolution │
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

## 3. Trained Model & Dataset Verification

### Dataset Ground Truth
The machine learning model was trained using the benchmark phishing corpus ([`zefang-liu/phishing-email-dataset`](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset) / `phishing_email.csv`).

- **Total Volume**: ~82,486 raw rows (~82,077 clean, deduplicated entries).
- **Features**: `text_combined` (subject and body text), `label`.
- **Empirical Label Semantics (Verified)**:
  - **`label = 0`**: **Legitimate** (Corporate communication from Enron, Ham corpora).
  - **`label = 1`**: **Phishing** (Malicious lures, spam, credential harvesting from Nazario, SpamAssassin, and CEAS).

### Model Pipeline Specifications
- **Persistence Path**: `models/phishing_model.joblib`
- **Vectorizer**: `TfidfVectorizer` (50,000 features, unigrams & bigrams, sublinear term frequency).
- **Classifier**: `LogisticRegression(C=1.5, class_weight='balanced', max_iter=1000)`.
- **Test Performance in Colab**: Approximately 98.82% accuracy on initial held-out split.
  *(Note: Held-out benchmark metrics do not guarantee 100% real-world detection against novel zero-day phishing lures).*

---

## 4. Technology Stack

- **Frontend**: HTML5, Vanilla CSS3 (Custom Dark Navy Theme), Vanilla JavaScript (ES6+), Fetch API.
- **Backend**: Python 3.10+, Flask, REST API.
- **Machine Learning**: Scikit-learn, TF-IDF Vectorization, Logistic Regression, Pandas, NumPy, Joblib.
- **Testing**: Pytest (40 passing tests across API, DFA, model inference, and edge cases).
- **Environment**: macOS Sonoma / Sequoia, Git.

---

## 5. Project Directory Structure

```
Phishing Email Detection/
├── app.py                      # Flask REST API server and routing
├── requirements.txt            # Python dependencies (Flask, scikit-learn, pandas, joblib, pytest)
├── README.md                   # Comprehensive documentation and operational guide
├── .gitignore                  # Git exclusions for .venv, pycache, and binary model artifacts
├── config.py                   # Central configuration (thresholds, security limits, scoring weights)
├── dfa_detector.py             # Formal DFA engine (PhraseDFA & URLStructureDFA)
├── model_detector.py           # ML inference engine and salient feature extractor
├── train_model.py              # ML training pipeline with anti-leakage stratified splitting
├── combined_detector.py        # Multi-modal fusion engine and risk scoring
├── slm_detector.py             # Modular Small Language Model interface
├── templates/
│   └── index.html              # Dark-themed cybersecurity dashboard template
├── static/
│   ├── style.css               # Vanilla CSS design system (Dark navy, indigo/cyan accents)
│   └── script.js               # Vanilla JS client (Fetch API, presets, safe textContent rendering)
├── data/
│   ├── README.md               # Dataset sourcing, format specs, and anti-leakage guide
│   └── sample_dataset.csv      # Curated dataset for training validation
├── models/
│   ├── .gitkeep                # Tracks models directory in Git
│   └── phishing_model.joblib   # Trained scikit-learn model artifact
├── tests/
│   ├── __init__.py             # Test package marker
│   ├── test_api.py             # API endpoint validation and error handling tests
│   ├── test_dfa.py             # DFA token transitions and URL state machine tests
│   ├── test_model.py           # ML training, inference, and missing-model tests
│   └── test_detection.py       # Comprehensive scenario, disagreement, and failure mode tests
└── docs/
    └── architecture.md         # Comprehensive system specification & mathematical DFA model
```

---

## 6. Prerequisites & macOS Setup

- **Operating System**: macOS (Apple Silicon or Intel).
- **Python**: Python 3.10, 3.11, 3.12, or 3.14.
- **Terminal**: `zsh` or `bash`.

Verify your Python installation:
```bash
python3 --version
```

---

## 7. Installation & Launch Instructions

### Step 1: Open Terminal in Workspace
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

### Step 4: Verify Model Placement
Ensure the trained model file `phishing_model.joblib` is located in the `models/` directory:
```bash
ls -lh models/phishing_model.joblib
```
*(If missing, copy your exported Colab model into `models/phishing_model.joblib` or run `python train_model.py` to train a baseline model).*

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

## 8. Automated Test Suite

PhishGuard includes **40 automated unit, integration, and scenario tests** covering:
- Ordinary benign emails.
- Urgent account verification lures and direct credential requests.
- URL character-level DFA checks (IPv4 hosts, userinfo `@` spoofing, non-standard ports, suspicious TLDs).
- Input boundary validation (empty input, whitespace-only, missing JSON fields, malformed JSON, oversized payloads > 50,000 chars).
- Model failure modes (missing file, corrupted weights).
- Disagreement handling between DFA rules and ML statistics.
- Verified label mapping (`0 = Legitimate`, `1 = Phishing`).

Run the full test suite:
```bash
pytest tests/ -v
```

Expected output:
```
tests/test_api.py::TestAPIEndpoints::test_get_index_page PASSED
tests/test_detection.py::TestEmailScenarios::test_simple_ordinary_email PASSED
tests/test_detection.py::TestEmailScenarios::test_urgent_account_verification_language PASSED
tests/test_detection.py::TestEmailScenarios::test_email_requesting_credentials PASSED
tests/test_detection.py::TestEmailScenarios::test_email_with_suspicious_url PASSED
tests/test_detection.py::TestModelFailureModesAndDisagreements::test_disagreement_between_dfa_and_ml PASSED
tests/test_detection.py::TestModelFailureModesAndDisagreements::test_verified_label_mapping PASSED
...
============================== 40 passed in 1.13s ==============================
```

---

## 9. REST API Documentation & Endpoints

### `GET /api/health`
Returns backend service status and component availability.

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
      "registered_phrases": 80,
      "status": "active"
    },
    "machine_learning_model": {
      "decision_threshold": 0.5,
      "is_loaded": true,
      "model_path": ".../models/phishing_model.joblib",
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
Analyzes email content for phishing indicators. Supports both unified string format and multi-field format.

**Headers**:
- `Content-Type: application/json`

**Unified Payload Format**:
```json
{
  "email_text": "URGENT: Your account has been suspended! You must confirm your password at http://192.168.1.50/auth within 24 hours."
}
```

**Multi-Field Payload Format**:
```json
{
  "subject": "URGENT: Your account has been suspended",
  "body": "Confirm your password at http://192.168.1.50/auth within 24 hours.",
  "sender": "service-alert@chase-support.xyz"
}
```

**Example Request:**
```bash
curl -s -X POST http://127.0.0.1:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"email_text": "URGENT: Your account has been suspended! You must confirm your password at http://192.168.1.50/auth within 24 hours."}'
```

**Example Response:**
```json
{
  "status": "success",
  "classification": "Potential Phishing",
  "final_classification": "PHISHING",
  "risk_level": "CRITICAL",
  "risk_score": 85.7,
  "operating_mode": "HYBRID_DFA_AND_ML",
  "is_reduced_capability": false,
  "model_prediction": "PHISHING",
  "model_confidence": 0.9741,
  "dfa_result": {
    "detected_phrases": [
      {
        "pattern": "account has been suspended",
        "category": "Account Threat",
        "severity": "HIGH"
      },
      {
        "pattern": "confirm your password",
        "category": "Credential Harvesting",
        "severity": "CRITICAL"
      }
    ],
    "url_findings": [
      {
        "url": "http://192.168.1.50/auth",
        "host": "192.168.1.50",
        "anomalies": [
          {
            "type": "IP_ADDRESS_HOST",
            "severity": "CRITICAL",
            "explanation": "Host is a raw numerical IP address (192.168.1.50) rather than a registered domain name."
          }
        ]
      }
    ],
    "dfa_risk_score": 95.0,
    "total_transitions": 47
  },
  "machine_learning": {
    "is_available": true,
    "raw_label": 1,
    "mapped_classification": "Phishing",
    "label_mapping_verified": true,
    "phishing_probability": 0.9741,
    "legitimate_probability": 0.0259,
    "confidence": 0.9741,
    "contributing_features": [
      {"term": "account", "direction": "Phishing", "weight": 0.8181},
      {"term": "http", "direction": "Phishing", "weight": 0.7815},
      {"term": "urgent", "direction": "Phishing", "weight": 0.3613}
    ]
  },
  "indicators": [
    "URL Anomaly [IP_ADDRESS_HOST]: Host is a raw numerical IP address...",
    "DFA Pattern [Account Threat]: 'account has been suspended'...",
    "DFA Pattern [Credential Harvesting]: 'confirm your password'..."
  ],
  "reasons": [
    "URL Anomaly [IP_ADDRESS_HOST]: Host is a raw numerical IP address...",
    "DFA Pattern [Account Threat]: 'account has been suspended'...",
    "Trained ML Classifier estimated phishing probability of 97.4%...",
    "Risk floor elevated to 75+ due to critical DFA security violations."
  ],
  "disclaimer": "Disclaimer: PhishGuard analysis results are heuristic and statistical estimates intended for educational and triage assistance. Automated classifications are not guaranteed to be 100% correct; always verify sensitive requests through out-of-band channels."
}
```

---

## 10. DFA State Machine Specification

Formally, the token-based pattern detector is modeled as a 5-tuple:
$$M = (Q, \Sigma, \delta, q_0, F)$$

- **States ($Q$)**: Set of prefix states representing recognized subsequences of multi-token phishing phrases.
- **Alphabet ($\Sigma$)**: Normalized lowercase word tokens.
- **Transition Function ($\delta$)**: Deterministic step function transitioning to subsequent prefix states on token consumption with pre-computed failure links for $O(N)$ single-pass processing without backtracking.
- **Accepting States ($F$)**: Terminal nodes signifying complete pattern detection with assigned severity weights (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`).

---

## 11. SLM Clarification & Future Integration

### Important Clarification
The machine learning classifier integrated into PhishGuard is a **TF-IDF + Logistic Regression statistical pipeline**, **not a Small Language Model (SLM)**. 

### Future SLM Integration Architecture
PhishGuard provides a dedicated modular interface in [`slm_detector.py`](file:///Users/sriramprasath/Documents/Phishing%20Email%20Detection/slm_detector.py) designed to host a local SLM (e.g., Llama-3.2-1B, Phi-3.5-mini, or Gemma-2-2B) to audit conversational intent and coercion:
1. **Prompt Isolation**: Treats email content as untrusted input enclosed within `<UNTRUSTED_EMAIL_CONTENT>` tags to resist prompt injection.
2. **Local Inference Requirements**: Running an SLM locally requires approximately 2 GB–4 GB of RAM/VRAM using an inference engine like Ollama or vLLM (`export PHISHGUARD_SLM_ENABLED=true`).
3. **No External Leaks**: Email contents are never transmitted to third-party proprietary APIs.

---

## 12. Limitations, Adversarial Threats & Privacy

- **Evasion via Novel Phrasing**: Highly conversational, bespoke spear-phishing messages lacking traditional spam vocabulary may yield lower statistical ML scores.
- **Obfuscation**: Intentional token misspellings or homoglyphs (e.g. `p@ssw0rd`) can bypass exact DFA token matches unless preprocessed with character normalizers.
- **Decision Support Only**: Automated classifications are triage aids and cannot replace human scrutiny and out-of-band verification.
- **Privacy First**: Submitted emails are analyzed in-memory and are never stored to persistent databases or logged to external services.

---

## 13. Git & GitHub Setup

The project includes `.gitignore` to prevent committing virtual environments (`.venv`), temporary cache files, or large joblib binaries.

To push changes to GitHub:
```bash
git add .
git commit -m "feat: complete PhishGuard with integrated Colab model, verified label mapping, and comprehensive test suite"
git push origin main
```
