"""PhishGuard Comprehensive Detection Test Suite.

Tests the full detection spectrum:
- Ordinary benign emails
- Urgent account verification lures
- Direct credential harvesting requests
- Suspicious URL anomalies (raw IP host, userinfo spoofing)
- Input validation (empty, whitespace, missing fields, invalid JSON, oversized input)
- Model failure modes (missing file, prediction errors)
- Disagreement handling between DFA and ML
- Verified label mapping (0 = Legitimate, 1 = Phishing)
"""

from pathlib import Path
import json
import pytest

from app import create_app
from config import TestingConfig
from model_detector import ModelDetector
from combined_detector import CombinedDetector
from dfa_detector import DFADetector


@pytest.fixture
def client():
    """Provides a Flask test client configured for testing."""
    test_app = create_app(TestingConfig)
    with test_app.test_client() as test_client:
        yield test_client


@pytest.fixture
def detector():
    """Provides an instantiated CombinedDetector."""
    return CombinedDetector()


class TestEmailScenarios:
    """Tests realistic email classification scenarios."""

    def test_simple_ordinary_email(self, detector):
        """Ordinary corporate email should be classified as Likely Legitimate with low risk."""
        ordinary_email = (
            "Hi Sarah, could you please send over the latest draft of the quarterly report? "
            "We have a meeting with the client on Thursday afternoon to review the progress."
        )
        res = detector.analyze_email(ordinary_email)
        assert res["status"] == "success"
        assert res["final_classification"] in ("LIKELY_LEGITIMATE", "SUSPICIOUS")
        assert res["risk_score"] < 50.0
        assert len(res["dfa"]["detected_phrases"]) == 0

    def test_urgent_account_verification_language(self, detector):
        """Email with urgent verification demands should trigger DFA and produce high risk."""
        urgent_email = (
            "URGENT: Immediate attention required! Your account has been suspended due to security violations. "
            "You must verify your account within 24 hours to prevent permanent deactivation."
        )
        res = detector.analyze_email(urgent_email)
        assert res["status"] == "success"
        assert res["risk_score"] >= 65.0
        assert res["final_classification"] == "PHISHING"
        assert res["classification"] == "Potential Phishing"
        
        patterns = [p["pattern"] for p in res["dfa"]["detected_phrases"]]
        assert any("account" in p and "suspended" in p for p in patterns)
        assert "verify your account" in patterns

    def test_email_requesting_credentials(self, detector):
        """Direct password confirmation request should trigger critical DFA rule."""
        credential_email = (
            "Security Alert: We detected an unauthorized sign-in attempt. "
            "Please confirm your password immediately to secure your credentials."
        )
        res = detector.analyze_email(credential_email)
        assert res["status"] == "success"
        assert res["risk_score"] >= 65.0
        patterns = [p["pattern"] for p in res["dfa"]["detected_phrases"]]
        assert "confirm your password" in patterns

    def test_email_with_suspicious_url(self, detector):
        """Email pointing to a raw IP host or spoofed authority must be flagged."""
        url_email = "Click here to access your statement: http://192.168.1.100/login"
        res = detector.analyze_email(url_email)
        assert res["status"] == "success"
        assert len(res["dfa"]["url_findings"]) > 0
        
        anomalies = res["dfa"]["url_findings"][0]["anomalies"]
        anomaly_types = [a["type"] for a in anomalies]
        assert "IP_ADDRESS_HOST" in anomaly_types


class TestAPIValidation:
    """Tests API error handling, boundaries, and validation."""

    def test_empty_input_rejected(self, client):
        response = client.post(
            "/api/analyze",
            data=json.dumps({"email_text": ""}),
            content_type="application/json"
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
        assert "empty email" in data["error"].lower()

    def test_whitespace_only_input_rejected(self, client):
        response = client.post(
            "/api/analyze",
            data=json.dumps({"email_text": "    \n\t   "}),
            content_type="application/json"
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_missing_json_fields_rejected(self, client):
        response = client.post(
            "/api/analyze",
            data=json.dumps({}),
            content_type="application/json"
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_invalid_json_rejected(self, client):
        response = client.post(
            "/api/analyze",
            data='{"email_text": broken_json_no_quotes}',
            content_type="application/json"
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_oversized_email_rejected(self, client):
        oversized = "A" * 50001
        response = client.post(
            "/api/analyze",
            data=json.dumps({"email_text": oversized}),
            content_type="application/json"
        )
        assert response.status_code == 413
        data = response.get_json()
        assert "exceeds maximum allowed length" in data["error"]


class TestModelFailureModesAndDisagreements:
    """Tests resilience during model failures and detector disagreements."""

    def test_model_loading_failure_handling(self):
        """Non-existent model path must return clear error and engage reduced capability mode."""
        dummy_path = Path("/tmp/non_existent_phishguard_model.joblib")
        detector = ModelDetector(model_path=dummy_path)
        
        assert not detector.is_loaded
        res = detector.predict_email("Some test email")
        assert res["is_available"] is False
        assert "not found" in res["status_message"].lower()

    def test_model_prediction_failure_handled_gracefully(self):
        """Corrupted pipeline object must return safe error status without crashing."""
        detector = ModelDetector()
        # Mock pipeline to raise an exception
        class CorruptedPipeline:
            def predict_proba(self, X):
                raise RuntimeError("Corrupted weights matrix")
        
        detector.pipeline = CorruptedPipeline()
        detector.is_loaded = True
        
        res = detector.predict_email("Test content")
        assert res["is_available"] is False
        assert "Inference execution error" in res["status_message"]

    def test_disagreement_between_dfa_and_ml(self):
        """When DFA detects critical credential harvesting but ML returns low score,
        the combined detector must note disagreement and enforce security floor."""
        dfa_mock = DFADetector()
        
        # Mock ML detector that returns 0% phishing probability
        class BenignMLMock:
            is_loaded = True
            threshold = 0.50
            def predict(self, subject="", body="", sender=""):
                return {
                    "is_available": True,
                    "raw_label": 0,
                    "prediction": "LEGITIMATE",
                    "mapped_classification": "Legitimate",
                    "phishing_probability": 0.05,
                    "legitimate_probability": 0.95,
                    "confidence": 0.95,
                    "contributing_features": []
                }
        
        combined = CombinedDetector(dfa_detector=dfa_mock, model_detector=BenignMLMock())
        email_with_threat = "Please confirm your password at http://192.168.1.1/auth immediately."
        res = combined.analyze_email(email_with_threat)

        # The system must flag the disagreement and elevate the risk floor
        assert len(res["warnings"]) > 0 or any("Disagreement" in r for r in res["reasons"])
        assert res["risk_score"] >= 75.0
        assert res["final_classification"] == "PHISHING"

    def test_verified_label_mapping(self):
        """Verifies that 0 maps to Legitimate and 1 maps to Phishing based on dataset ground truth."""
        assert ModelDetector.VERIFIED_LABEL_MAP[0] == "Legitimate"
        assert ModelDetector.VERIFIED_LABEL_MAP[1] == "Phishing"
