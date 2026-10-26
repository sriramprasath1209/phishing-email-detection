"""PhishGuard Machine Learning Model Detector Module.

Responsible for safely loading the trained scikit-learn Pipeline (TF-IDF + Logistic Regression),
executing inference on email content, computing class probabilities, extracting contributing
salient features, and handling missing model states cleanly.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import os

import joblib
import numpy as np

from config import Config


class ModelDetector:
    """Inference engine for the custom-trained phishing classifier."""

    def __init__(self, model_path: Optional[Path] = None, threshold: float = 0.50):
        self.model_path = model_path or Config.MODEL_PATH
        self.threshold = threshold
        self.pipeline = None
        self.is_loaded = False
        self.load_error: Optional[str] = None
        self.load_model()

    def load_model(self) -> bool:
        """Attempts to load the serialized model pipeline from disk."""
        if not self.model_path.exists():
            self.is_loaded = False
            self.pipeline = None
            self.load_error = f"Model file not found at {self.model_path}."
            return False

        try:
            self.pipeline = joblib.load(self.model_path)
            self.is_loaded = True
            self.load_error = None
            return True
        except Exception as exc:
            self.is_loaded = False
            self.pipeline = None
            self.load_error = f"Failed to deserialize model: {str(exc)}"
            return False

    def predict(self, subject: str, body: str, sender: str = "") -> Dict[str, Any]:
        """Runs model inference on the provided email text.
        
        Returns:
            Dict containing availability status, predicted class, class probabilities,
            confidence score, decision threshold, salient keywords, and diagnostic message.
        """
        combined_text = f"{subject}\n{body}".strip()

        # If model is unavailable, return an explicit unavailable status (NOT a fake prediction)
        if not self.is_loaded or self.pipeline is None:
            return {
                "is_available": False,
                "prediction": None,
                "phishing_probability": None,
                "legitimate_probability": None,
                "confidence": None,
                "decision_threshold": self.threshold,
                "status_message": (
                    "Trained ML model unavailable. System operating in reduced-capability DFA mode. "
                    "Execute `python train_model.py` to train and activate the custom classifier."
                ),
                "contributing_features": []
            }

        if not combined_text:
            return {
                "is_available": True,
                "prediction": "INCONCLUSIVE",
                "phishing_probability": 0.0,
                "legitimate_probability": 1.0,
                "confidence": 0.0,
                "decision_threshold": self.threshold,
                "status_message": "Empty email text supplied to model.",
                "contributing_features": []
            }

        try:
            # Predict probabilities
            probabilities = self.pipeline.predict_proba([combined_text])[0]
            prob_legit = float(probabilities[0])
            prob_phish = float(probabilities[1])

            # Apply configurable decision threshold
            is_phishing = prob_phish >= self.threshold
            prediction = "PHISHING" if is_phishing else "LEGITIMATE"
            confidence = prob_phish if is_phishing else prob_legit

            # Extract salient contributing features (top n-grams)
            contributing_features = self._extract_salient_features(combined_text)

            return {
                "is_available": True,
                "prediction": prediction,
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
                "prediction": None,
                "phishing_probability": None,
                "legitimate_probability": None,
                "confidence": None,
                "decision_threshold": self.threshold,
                "status_message": f"Inference execution error: {str(exc)}",
                "contributing_features": []
            }

    def _extract_salient_features(self, text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Identifies words or bigrams from input text that contributed most heavily to the score."""
        try:
            tfidf = self.pipeline.named_steps["tfidf"]
            clf = self.pipeline.named_steps["clf"]

            feature_names = np.array(tfidf.get_feature_names_out())
            coefs = clf.coef_[0]

            # Vectorize current text
            vec = tfidf.transform([text]).toarray()[0]
            active_indices = np.where(vec > 0)[0]

            if len(active_indices) == 0:
                return []

            # Contribution = TF-IDF weight * logistic regression coefficient
            contributions = [
                {
                    "term": feature_names[idx],
                    "weight": round(float(vec[idx] * coefs[idx]), 4),
                    "direction": "Phishing" if coefs[idx] > 0 else "Legitimate"
                }
                for idx in active_indices
            ]

            # Sort by absolute impact
            contributions.sort(key=lambda x: abs(x["weight"]), reverse=True)
            return contributions[:top_k]
        except Exception:
            return []
