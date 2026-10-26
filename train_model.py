"""PhishGuard Custom Machine Learning Training Pipeline.

Trains a TF-IDF + Logistic Regression binary classifier to detect phishing emails.
Features:
- Automated deduplication and text normalization to prevent dataset leakage
- Stratified train/test splitting with fixed random seeds for reproducibility
- Scikit-learn Pipeline architecture (TF-IDF vectorizer fit strictly on train split)
- Exhaustive evaluation reporting: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- Model persistence via joblib
"""

import argparse
from pathlib import Path
import sys
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import joblib

from config import Config


def load_and_preprocess_dataset(csv_path: Path) -> Tuple[pd.Series, pd.Series]:
    """Loads CSV, validates columns, deduplicates, and cleans text samples."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # Validate column structure (support 'text' or 'text_combined')
    text_col = "text_combined" if "text_combined" in df.columns else "text"
    if text_col not in df.columns or "label" not in df.columns:
        raise ValueError(f"Dataset must contain 'label' and either 'text' or 'text_combined' columns. Found: {list(df.columns)}")

    # Clean missing values
    initial_count = len(df)
    df = df.dropna(subset=[text_col, "label"])

    # Cast and clean labels
    df["label"] = df["label"].astype(int)
    valid_labels = {0, 1}
    if not set(df["label"].unique()).issubset(valid_labels):
        raise ValueError(f"Labels must be 0 (legitimate) or 1 (phishing). Found: {df['label'].unique()}")

    # Anti-leakage: Deduplicate identical email texts
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df.drop_duplicates(subset=[text_col])
    deduped_count = len(df)

    print(f"[*] Loaded dataset: {initial_count} raw rows -> {deduped_count} deduplicated rows.")
    print(f"[*] Class distribution: {df['label'].value_counts().to_dict()} (0=Ham/Legitimate, 1=Phish)")

    if deduped_count < 10:
        raise ValueError("Dataset is too small to split and train meaningfully (minimum 10 rows required).")

    return df[text_col], df["label"]


def build_pipeline() -> Pipeline:
    """Builds a scikit-learn Pipeline encapsulating TF-IDF and Logistic Regression."""
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=4000,
                sublinear_tf=True,
                token_pattern=r"(?u)\b[a-zA-Z0-9']+\b"
            )
        ),
        (
            "classifier",
            LogisticRegression(
                C=1.5,
                max_iter=1000,
                random_state=42,
                class_weight="balanced"
            )
        )
    ])


def train_and_evaluate(
    data_path: Path,
    output_path: Path,
    test_size: float = 0.25,
    random_state: int = 42
) -> Dict[str, Any]:
    """Executes the complete training, evaluation, and serialization workflow."""
    print(f"\n==================================================")
    print(f"   PhishGuard Model Training Pipeline Starting   ")
    print(f"==================================================")
    print(f"Dataset source: {data_path}")
    print(f"Model destination: {output_path}")

    X, y = load_and_preprocess_dataset(data_path)

    # Stratified train/test split to preserve legitimate/phishing ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"[*] Training samples: {len(X_train)} | Test samples: {len(X_test)}")

    # Fit pipeline exclusively on training data
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Evaluate on held-out test split
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("\n---------------- Evaluation Metrics ----------------")
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")
    print(f"Confusion Matrix (rows=actual [0,1], cols=pred [0,1]):")
    print(f"  TN: {cm[0][0]:3d} | FP: {cm[0][1]:3d}")
    print(f"  FN: {cm[1][0]:3d} | TP: {cm[1][1]:3d}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"], digits=4))

    # Persist model
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    print(f"[+] Successfully saved trained model pipeline to: {output_path}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "output_path": str(output_path)
    }


def main():
    """Command-line interface entry point."""
    parser = argparse.ArgumentParser(description="Train the PhishGuard Phishing Detection ML Model")
    parser.add_argument(
        "--data",
        type=Path,
        default=Config.BASE_DIR / "data" / "sample_dataset.csv",
        help="Path to training dataset CSV (default: data/sample_dataset.csv)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Config.MODEL_PATH,
        help="Path to output saved joblib model (default: models/phishing_model.joblib)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Test split ratio (default: 0.25)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    try:
        train_and_evaluate(args.data, args.output, test_size=args.test_size, random_state=args.seed)
    except Exception as exc:
        print(f"\n[!] Training failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
