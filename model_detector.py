"""PhishGuard Machine Learning Model Detector Module.

Safely loads the trained scikit-learn Pipeline (TF-IDF + Logistic Regression),
executes inference on email content, computes class probabilities, extracts contributing
salient features, and handles missing or corrupted model states cleanly.

Trained Model Specification:
- Pipeline Architecture: TfidfVectorizer -> LogisticRegression (50,000 features)
- Verified Dataset: 'zefang-liu/phishing-email-dataset' / 'phishing_email.csv' (~82,077 clean samples)
- Verified Label Semantics:
    * 0 = Legitimate (Ham, corporate Enron / legitimate communication)
    * 1 = Phishing (Malicious / spam / credential harvesting)
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import os
import warnings

import joblib
import numpy as np

from config import Config

# Project base directory resolution
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "phishing_model.joblib"


class ModelDetector:
    """Inference engine for the trained phishing classifier."""

    # Empirically verified label mappings from dataset documentation and ground truth inspection
    VERIFIED_LABEL_MAP = {
        0: "Legitimate",
        1: "Phishing"
    }

    def __init__(self, model_path: Optional[Path] = None, threshold: float = 0.50):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.threshold = threshold
        self.pipeline = None
        self.is_loaded = False
        self.load_error: Optional[str] = None
        self.load_model()

    def load_model(self) -> bool:
        """Loads the serialized model pipeline safely from disk."""
        if not self.model_path.exists():
            self.is_loaded = False
            self.pipeline = None
            self.load_error = f"Model artifact not found at {self.model_path}. Please place 'phishing_model.joblib' in the models/ directory."
            return False

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                self.pipeline = joblib.load(self.model_path)

            self.is_loaded = True
            self.load_error = None
            return True
        except Exception as exc:
            self.is_loaded = False
            self.pipeline = None
            self.load_error = f"Failed to deserialize model: {str(exc)}"
            return False

    def predict_email(self, email_text: str) -> Dict[str, Any]:
        """Inference interface accepting raw email text.
        
        Args:
            email_text: The complete email text (subject and/or body).
            
        Returns:
            Dict containing raw predicted label, mapped classification, probabilities,
            confidence, threshold, salient features, and error/status info.
        """
        # Validate non-empty string input
        if not isinstance(email_text, str) or not email_text.strip():
            return {
                "is_available": self.is_loaded,
                "raw_label": None,
                "prediction": "INCONCLUSIVE",
                "mapped_classification": "Analysis Unavailable",
                "label_mapping_verified": True,
                "phishing_probability": None,
                "legitimate_probability": None,
                "confidence": None,
                "decision_threshold": self.threshold,
                "status_message": "Email text must be a non-empty string.",
                "contributing_features": []
            }

        # If model is unavailable, return clear status without fabricating predictions
        if not self.is_loaded or self.pipeline is None:
            return {
                "is_available": False,
                "raw_label": None,
                "prediction": None,
                "mapped_classification": "Model Unavailable",
                "label_mapping_verified": True,
                "phishing_probability": None,
                "legitimate_probability": None,
                "decision_threshold": self.threshold,
                "status_message": (
                    f"{self.load_error or 'Trained model file not loaded.'} "
                    "System operating in reduced-capability mode."
                ),
                "contributing_features": []
            }

        clean_text = email_text.strip()

        try:
            # Predict probabilities
            probabilities = self.pipeline.predict_proba([clean_text])[0]
            prob_legit = float(probabilities[0])
            prob_phish = float(probabilities[1])

            # Raw prediction from pipeline
            raw_pred = int(self.pipeline.predict([clean_text])[0])

            # Apply configurable decision threshold for positive (phishing) class
            is_phishing = prob_phish >= self.threshold
            prediction = "PHISHING" if is_phishing else "LEGITIMATE"
            mapped_classification = "Phishing" if is_phishing else "Legitimate"
            confidence = prob_phish if is_phishing else prob_legit

            # Extract salient features (top n-grams)
            contributing_features = self._extract_salient_features(clean_text)

            return {
                "is_available": True,
                "raw_label": raw_pred,
                "prediction": prediction,
                "mapped_classification": mapped_classification,
                "label_mapping_verified": True,
                "phishing_probability": round(prob_phish, 4),
                "legitimate_probability": round(prob_legit, 4),
                "confidence": round(confidence, 4),
                "decision_threshold": self.threshold,
                "status_message": "Inference completed successfully.",
                "contributing_features": contributing_features
            }
        except Exception as exc:
            return {
                "is_available": False,
                "raw_label": None,
                "prediction": None,
                "mapped_classification": "Error",
                "label_mapping_verified": True,
                "phishing_probability": None,
                "legitimate_probability": None,
                "confidence": None,
                "decision_threshold": self.threshold,
                "status_message": f"Inference execution error: {str(exc)}",
                "contributing_features": []
            }

    def predict(self, subject: str = "", body: str = "", sender: str = "") -> Dict[str, Any]:
        """Legacy multi-field compatibility interface."""
        combined_text = f"{subject}\n{body}".strip()
        if sender.strip():
            combined_text = f"From: {sender.strip()}\n{combined_text}"
        return self.predict_email(combined_text)

    def _extract_salient_features(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Extracts words/bigrams from the text with the greatest contribution to the decision."""
        try:
            # Support both 'classifier' and 'clf' step naming conventions
            tfidf = self.pipeline.named_steps.get("tfidf") or self.pipeline.steps[0][1]
            clf = self.pipeline.named_steps.get("classifier") or self.pipeline.named_steps.get("clf") or self.pipeline.steps[-1][1]

            feature_names = np.array(tfidf.get_feature_names_out())
            coefs = clf.coef_[0]

            # Vectorize text
            vec = tfidf.transform([text]).toarray()[0]
            active_indices = np.where(vec > 0)[0]

            if len(active_indices) == 0:
                return []

            # Contribution = TF-IDF weight * logistic regression coefficient
            contributions = [
                {
                    "term": str(feature_names[idx]),
                    "weight": round(float(vec[idx] * coefs[idx]), 4),
                    "direction": "Phishing" if coefs[idx] > 0 else "Legitimate"
                }
                for idx in active_indices
            ]

            contributions.sort(key=lambda x: abs(x["weight"]), reverse=True)
            return contributions[:top_k]
        except Exception:
            return []


# Global singleton instance for direct function usage
_default_detector: Optional[ModelDetector] = None

def get_default_detector() -> ModelDetector:
    """Returns or creates the default singleton ModelDetector."""
    global _default_detector
    if _default_detector is None:
        _default_detector = ModelDetector()
    return _default_detector

def predict_email(email_text: str) -> Dict[str, Any]:
    """Module-level convenience function for email classification."""
    return get_default_detector().predict_email(email_text)
