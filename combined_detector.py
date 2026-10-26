"""PhishGuard Combined Detection Engine.

Fuses multi-modal detection signals:
1. Deterministic Finite Automaton (Token-based Phrase DFA and URL Structure DFA)
2. Custom Machine Learning Classifier (TF-IDF + Logistic Regression)
3. Optional Small Language Model (SLM intent/coercion interface)

Combines findings transparently into an interpretable Risk Score (0-100),
evaluates detector disagreements, and maps to final verified classifications.
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

    def analyze_email(self, email_text: str) -> Dict[str, Any]:
        """Analyzes a single combined email text string."""
        return self.analyze(subject="", body=email_text)

    def analyze(self, subject: str = "", body: str = "", sender: str = "") -> Dict[str, Any]:
        """Runs integrated multi-engine analysis across email inputs."""
        clean_subject = (subject or "").strip()
        clean_body = (body or "").strip()
        clean_sender = (sender or "").strip()

        combined_text = f"{clean_subject}\n{clean_body}".strip()
        if clean_sender:
            combined_text = f"From: {clean_sender}\n{combined_text}"

        # Handle empty inputs
        if not combined_text:
            return {
                "status": "error",
                "classification": "Analysis Unavailable",
                "final_classification": "INCONCLUSIVE",
                "risk_level": "UNKNOWN",
                "risk_score": 0.0,
                "operating_mode": "IDLE",
                "is_reduced_capability": not self.model.is_loaded,
                "model_prediction": None,
                "model_confidence": None,
                "dfa_result": self.dfa.analyze("", ""),
                "indicators": [],
                "warnings": ["No email text provided for analysis."],
                "reasons": ["No email subject or body text provided for analysis."],
                "explanation": "No text provided. Please enter an email subject or body to analyze.",
                "dfa": self.dfa.analyze("", ""),
                "machine_learning": self.model.predict("", ""),
                "slm": self.slm.analyze("", ""),
                "disclaimer": self._get_disclaimer()
            }

        # 1. Execute DFA Scan
        dfa_result = self.dfa.analyze(clean_subject, clean_body, clean_sender)

        # 2. Execute ML Inference
        ml_result = self.model.predict(clean_subject, clean_body, clean_sender)

        # 3. Query Optional SLM Interface
        slm_result = self.slm.analyze(clean_subject, clean_body, clean_sender)

        # 4. Synthesize Evidence, Warnings, and Rationale
        reasons: List[str] = []
        warnings_list: List[str] = []
        indicators: List[str] = []

        dfa_score = dfa_result["dfa_risk_score"]
        url_findings = dfa_result.get("url_findings", [])
        detected_phrases = dfa_result.get("detected_phrases", [])

        # Collect URL anomalies
        url_anomalies_count = 0
        critical_url_flags = False
        for u in url_findings:
            for anom in u.get("anomalies", []):
                url_anomalies_count += 1
                flag_str = f"URL Anomaly [{anom['type']}]: {anom['explanation']}"
                indicators.append(flag_str)
                reasons.append(flag_str)
                if anom.get("severity") == "CRITICAL":
                    critical_url_flags = True

        # Collect DFA phrase matches
        critical_phrase_flags = False
        for p in detected_phrases:
            phrase_str = f"DFA Pattern [{p['category']}]: '{p['pattern']}' ({p['explanation']})"
            indicators.append(phrase_str)
            reasons.append(phrase_str)
            if p.get("severity") == "CRITICAL":
                critical_phrase_flags = True

        # Decision synthesis
        if ml_result["is_available"] and ml_result["phishing_probability"] is not None:
            operating_mode = "HYBRID_DFA_AND_ML"
            is_reduced = False
            ml_prob = ml_result["phishing_probability"]
            ml_pred = ml_result["prediction"]
            ml_score = ml_prob * 100.0

            reasons.append(
                f"Trained ML Classifier estimated phishing probability of {ml_prob * 100:.1f}% "
                f"(Threshold: {self.model.threshold * 100:.0f}%, Class: {ml_result.get('mapped_classification', ml_pred)})."
            )

            # Highlight salient features
            top_features = ml_result.get("contributing_features", [])
            if top_features:
                feature_summary = ", ".join(f"'{f['term']}' ({f['direction']})" for f in top_features[:3])
                reasons.append(f"Salient model features: {feature_summary}.")

            # Check detector agreement/disagreement
            dfa_is_phish = (dfa_score >= 35.0 or critical_phrase_flags or critical_url_flags)
            ml_is_phish = (ml_pred == "PHISHING")

            if dfa_is_phish and not ml_is_phish:
                # DFA found concrete threat patterns, but ML fell below threshold
                disagree_msg = (
                    "Detector Disagreement: DFA matched explicit threat indicators (e.g., credential requests or IP URLs), "
                    "while statistical ML returned a lower probability. Conservative policy elevates risk to protect against credential theft."
                )
                warnings_list.append(disagree_msg)
                reasons.append(disagree_msg)

            elif ml_is_phish and not dfa_is_phish:
                # ML found vocabulary correlation, but no exact DFA phrases matched
                disagree_msg = (
                    "Detector Notice: ML classifier identified suspicious vocabulary correlation, "
                    "though no configured exact DFA phrase patterns were triggered."
                )
                reasons.append(disagree_msg)

            # Multi-factor score formulation
            url_penalty = min(100.0, url_anomalies_count * 25.0)
            combined_score = (
                (ml_score * getattr(self.config, "WEIGHT_ML_MODEL", 0.50)) +
                (dfa_score * getattr(self.config, "WEIGHT_DFA", 0.35)) +
                (url_penalty * getattr(self.config, "WEIGHT_STRUCTURAL_URL", 0.15))
            )

            # Security floor: Critical DFA indicators guarantee high-risk verdict
            if critical_url_flags or critical_phrase_flags:
                combined_score = max(combined_score, 75.0)
                reasons.append("Risk floor elevated to 75+ due to critical DFA security violations.")

            risk_score = min(100.0, max(0.0, round(combined_score, 1)))

        else:
            # Reduced Capability Mode: ML model file missing
            operating_mode = "REDUCED_CAPABILITY_DFA_ONLY"
            is_reduced = True
            warn_msg = (
                "Operating in Reduced Capability Mode: Trained ML model is unavailable. "
                "Analysis is based strictly on DFA rules and URL heuristics."
            )
            warnings_list.append(warn_msg)
            reasons.append(warn_msg)

            url_penalty = min(100.0, url_anomalies_count * 30.0)
            combined_score = (dfa_score * 0.70) + (url_penalty * 0.30)
            if critical_url_flags or critical_phrase_flags:
                combined_score = max(combined_score, 70.0)
            risk_score = min(100.0, max(0.0, round(combined_score, 1)))

        # Determine Final Classification and Risk Level
        threshold_suspicious = getattr(self.config, "THRESHOLD_SUSPICIOUS", 65.0)
        threshold_legit = getattr(self.config, "THRESHOLD_LEGITIMATE", 30.0)

        if is_reduced and dfa_score == 0 and url_anomalies_count == 0:
            final_classification = "INCONCLUSIVE"
            classification_label = "Analysis Unavailable"
            risk_level = "LOW_CONFIDENCE"
            reasons.append(
                "No known malicious patterns detected by DFA, but cannot confirm legitimacy "
                "because machine learning model is inactive."
            )
            explanation = "No malicious DFA indicators matched, but machine learning validation was unavailable."
        elif risk_score >= threshold_suspicious:
            final_classification = "PHISHING"
            classification_label = "Potential Phishing"
            risk_level = "CRITICAL" if risk_score >= 80.0 else "HIGH"
            explanation = f"Classified as Potential Phishing with high risk score ({risk_score:.0f}/100) based on multi-factor analysis."
        elif risk_score >= threshold_legit:
            final_classification = "SUSPICIOUS"
            classification_label = "Suspicious"
            risk_level = "MODERATE"
            explanation = f"Classified as Suspicious with moderate risk score ({risk_score:.0f}/100); exercise caution before interacting."
        else:
            final_classification = "LIKELY_LEGITIMATE"
            classification_label = "Likely Legitimate"
            risk_level = "LOW"
            reasons.append("Email text exhibits benign statistical and syntactic characteristics.")
            explanation = f"Classified as Likely Legitimate ({risk_score:.0f}/100); no significant threat indicators identified."

        return {
            "status": "success",
            "classification": classification_label,
            "final_classification": final_classification,
            "risk_level": risk_level,
            "risk_score": risk_score,
            "operating_mode": operating_mode,
            "is_reduced_capability": is_reduced,
            "model_prediction": ml_result.get("prediction"),
            "model_confidence": ml_result.get("confidence"),
            "dfa_result": dfa_result,
            "indicators": indicators,
            "explanation": explanation,
            "reasons": reasons,
            "warnings": warnings_list,
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
