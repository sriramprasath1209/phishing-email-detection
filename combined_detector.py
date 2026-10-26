"""PhishGuard Combined Detection Engine.

Fuses multi-modal detection signals:
1. Deterministic Finite Automaton (Phrase matching and URL structural analysis)
2. Custom Machine Learning Classifier (TF-IDF + Logistic Regression)
3. Optional Small Language Model (SLM intent/coercion interface)

Combines findings transparently into an interpretable Risk Score (0-100) and
Final Classification, with fallback into clearly labeled Reduced Capability Mode
if the ML model or SLM is unavailable.
"""

from typing import Dict, Any, List, Optional
from config import Config
from dfa_detector import DFADetector
from model_detector import ModelDetector
from slm_detector import SLMDetector


class CombinedDetector:
    """Master engine orchestrating DFA, ML, and SLM analysis pipelines."""

    def __init__(
        self,
        config: Optional[Config] = None,
        dfa_detector: Optional[DFADetector] = None,
        model_detector: Optional[ModelDetector] = None,
        slm_detector: Optional[SLMDetector] = None
    ):
        if config is None:
            self.config = Config()
        elif isinstance(config, type):
            self.config = config()
        else:
            self.config = config

        model_path = getattr(self.config, "MODEL_PATH", Config.MODEL_PATH)
        threshold = getattr(self.config, "MODEL_DECISION_THRESHOLD", Config.MODEL_DECISION_THRESHOLD)

        self.dfa = dfa_detector or DFADetector(self.config)
        self.model = model_detector or ModelDetector(model_path, threshold)
        self.slm = slm_detector or SLMDetector(self.config)

    def analyze(self, subject: str, body: str, sender: str = "") -> Dict[str, Any]:
        """Runs integrated multi-engine analysis across the email inputs."""
        clean_subject = (subject or "").strip()
        clean_body = (body or "").strip()
        clean_sender = (sender or "").strip()

        # Handle empty inputs
        if not clean_subject and not clean_body:
            return {
                "final_classification": "INCONCLUSIVE",
                "risk_level": "UNKNOWN",
                "risk_score": 0.0,
                "operating_mode": "IDLE",
                "is_reduced_capability": not self.model.is_loaded,
                "reasons": ["No email subject or body text provided for analysis."],
                "dfa": self.dfa.analyze("", ""),
                "machine_learning": self.model.predict("", ""),
                "slm": self.slm.analyze("", ""),
                "disclaimer": self._get_disclaimer()
            }

        # 1. Execute DFA Scan
        dfa_result = self.dfa.analyze(clean_subject, clean_body, clean_sender)

        # 2. Execute ML Inference
        ml_result = self.model.predict(clean_subject, clean_body, clean_sender)

        # 3. Query Optional SLM
        slm_result = self.slm.analyze(clean_subject, clean_body, clean_sender)

        # 4. Synthesize Evidence and Calculate Risk Score
        reasons: List[str] = []
        dfa_score = dfa_result["dfa_risk_score"]
        url_findings = dfa_result.get("url_findings", [])
        detected_phrases = dfa_result.get("detected_phrases", [])

        # Collect URL anomalies
        url_anomalies_count = 0
        critical_url_flags = False
        for u in url_findings:
            for anom in u.get("anomalies", []):
                url_anomalies_count += 1
                if anom.get("severity") == "CRITICAL":
                    critical_url_flags = True
                reasons.append(f"URL Anomaly [{anom['type']}]: {anom['explanation']}")

        # Collect DFA phrase matches
        critical_phrase_flags = False
        for p in detected_phrases:
            if p.get("severity") == "CRITICAL":
                critical_phrase_flags = True
            reasons.append(f"DFA Pattern [{p['category']}]: '{p['pattern']}' ({p['explanation']})")

        # Determine operating mode & calculate risk score
        if ml_result["is_available"] and ml_result["phishing_probability"] is not None:
            operating_mode = "HYBRID_DFA_AND_ML"
            is_reduced = False
            ml_prob = ml_result["phishing_probability"]
            ml_score = ml_prob * 100.0

            reasons.append(
                f"ML Classifier estimated phishing probability of {ml_prob * 100:.1f}% "
                f"(Decision Threshold: {self.config.MODEL_DECISION_THRESHOLD * 100:.0f}%)."
            )

            # Feature explanation from model
            top_features = ml_result.get("contributing_features", [])
            if top_features:
                feature_summary = ", ".join(f"'{f['term']}' ({f['direction']})" for f in top_features[:3])
                reasons.append(f"Salient model features influencing classification: {feature_summary}.")

            # Transparent weighted combination:
            # 50% ML Probability + 35% DFA Phrase Score + 15% URL Structural Weight
            url_penalty = min(100.0, url_anomalies_count * 25.0)
            combined_score = (
                (ml_score * self.config.WEIGHT_ML_MODEL) +
                (dfa_score * self.config.WEIGHT_DFA) +
                (url_penalty * self.config.WEIGHT_STRUCTURAL_URL)
            )

            # Safety floor: If high-severity DFA indicators exist (e.g. raw IP host or password harvest),
            # ensure risk score reflects genuine threat regardless of low statistical confidence
            if critical_url_flags or critical_phrase_flags:
                combined_score = max(combined_score, 75.0)
                reasons.append("Risk floor elevated to 75+ due to critical DFA security violations.")

            risk_score = min(100.0, max(0.0, round(combined_score, 1)))

        else:
            # Reduced Capability Mode: ML model is absent
            operating_mode = "REDUCED_CAPABILITY_DFA_ONLY"
            is_reduced = True
            reasons.append(
                "Operating in Reduced Capability Mode: Custom ML model is not loaded. "
                "Scoring is based strictly on DFA rules and URL heuristics."
            )

            # Calculate heuristic score from DFA
            url_penalty = min(100.0, url_anomalies_count * 30.0)
            combined_score = (dfa_score * 0.70) + (url_penalty * 0.30)
            if critical_url_flags or critical_phrase_flags:
                combined_score = max(combined_score, 70.0)
            risk_score = min(100.0, max(0.0, round(combined_score, 1)))

        # Determine Final Classification and Risk Level
        if is_reduced and dfa_score == 0 and url_anomalies_count == 0:
            # Rule: Do not label an email "safe" solely because no configured patterns matched
            # when the ML classifier is unavailable
            final_classification = "INCONCLUSIVE"
            risk_level = "LOW_CONFIDENCE"
            reasons.append(
                "No known malicious patterns detected by DFA, but cannot confirm legitimacy "
                "because machine learning model is inactive."
            )
        elif risk_score >= self.config.THRESHOLD_SUSPICIOUS:
            final_classification = "PHISHING"
            risk_level = "CRITICAL" if risk_score >= 80.0 else "HIGH"
        elif risk_score >= self.config.THRESHOLD_LEGITIMATE:
            final_classification = "SUSPICIOUS"
            risk_level = "MODERATE"
        else:
            final_classification = "LIKELY_LEGITIMATE"
            risk_level = "LOW"
            reasons.append("Email text exhibits benign statistical and syntactic characteristics.")

        return {
            "final_classification": final_classification,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "operating_mode": operating_mode,
            "is_reduced_capability": is_reduced,
            "reasons": reasons,
            "dfa": dfa_result,
            "machine_learning": ml_result,
            "slm": slm_result,
            "disclaimer": self._get_disclaimer()
        }

    @staticmethod
    def _get_disclaimer() -> str:
        return (
            "Disclaimer: PhishGuard analysis results are heuristic and statistical estimates "
            "intended for educational and triage assistance. Automated classifications are not "
            "guaranteed to be 100% correct; always verify sensitive requests through out-of-band channels."
        )
