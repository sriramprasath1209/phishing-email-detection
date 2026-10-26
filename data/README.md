# PhishGuard Dataset Guide

This directory contains email dataset files and documentation for training the PhishGuard machine learning classifier.

## Dataset Format Specification

The training pipeline expects a CSV file formatted as follows:

| Column Header | Data Type | Description | Values / Examples |
| :--- | :--- | :--- | :--- |
| `text` | String | Normalized email content (Subject + Body) | "Urgent: Verify your account immediately..." |
| `label` | Integer | Binary classification ground truth | `0` = Legitimate (Ham), `1` = Phishing |

### Sample CSV Structure:
```csv
text,label
"Team sync meeting scheduled for Thursday at 2 PM to review quarterly OKRs.",0
"Your account has been suspended due to suspicious activity. Verify credentials at http://192.168.1.1/auth now.",1
```

## Included Starter Dataset

PhishGuard provides `data/sample_dataset.csv` containing curated, realistic legitimate and phishing email examples across diverse corporate, personal, financial, and administrative categories. This allows developers and evaluators to train, test, and validate the machine learning pipeline immediately without requiring external downloads.

## Using Large-Scale External Datasets

To train a production-grade classifier on tens of thousands of emails, you can download public research corpora:

1. **Nazario Phishing Corpus**:
   - Contains thousands of verified real-world phishing emails collected since 2005.
   - Available via [Monkey.org Phishing Corpus](https://monkey.org/~jose/phishing/).

2. **Enron Ham Dataset**:
   - Extensive corpus of legitimate corporate communication.
   - Available via [CMU Enron Email Dataset](https://www.cs.cmu.edu/~enron/).

3. **Kaggle Phishing Email Datasets**:
   - Aggregated collections such as `Phishing Email Dataset` (Subhajit Nayak) or `SpamAssassin Public Mail Corpus`.

### Training with an External Dataset
Run the training script pointing to your custom CSV:
```bash
python train_model.py --data /path/to/custom_dataset.csv --output models/phishing_model.joblib
```

## Data Hygiene & Anti-Leakage Measures
- **Deduplication**: `train_model.py` automatically detects and removes duplicate email texts before splitting to prevent training data leaking into the test partition.
- **Stratified Splitting**: Ensures class proportions (legitimate vs. phishing) are preserved identically across train and test sets.
- **Scikit-Learn Pipeline**: The TF-IDF vectorizer is fitted **exclusively** on the training split to eliminate feature leakage.
