"""Automated tests for the PhishGuard Machine Learning Pipeline and Inference Module."""

from pathlib import Path
import tempfile
import pytest

from config import Config
from model_detector import ModelDetector
from train_model import train_and_evaluate, load_and_preprocess_dataset
from combined_detector import CombinedDetector


class TestModelDetectorMissingState:
    """Tests model detector behavior when model file does not exist."""

    def test_missing_model_returns_graceful_status(self):
        non_existent_path = Path("/tmp/non_existent_phishguard_model.joblib")
        detector = ModelDetector(model_path=non_existent_path)

        assert not detector.is_loaded
        assert detector.pipeline is None

        res = detector.predict("Subject", "Body content")
        assert res["is_available"] is False
        assert res["prediction"] is None
        assert res["phishing_probability"] is None
        assert "reduced-capability" in res["status_message"].lower()

    def test_combined_detector_reduced_capability_mode(self):
        non_existent_path = Path("/tmp/non_existent_phishguard_model.joblib")
        detector = ModelDetector(model_path=non_existent_path)
        combined = CombinedDetector(model_detector=detector)

        # Analysis on an email with suspicious DFA keywords
        res = combined.analyze("URGENT: Verify your account", "Confirm your password at http://192.168.1.1/auth")
        assert res["is_reduced_capability"] is True
        assert res["operating_mode"] == "REDUCED_CAPABILITY_DFA_ONLY"
        assert res["risk_score"] > 50.0

        # Analysis on a benign email without ML loaded should NOT be blindly marked 'safe'
        benign_res = combined.analyze("Team lunch", "Let's grab burritos at noon.")
        assert benign_res["is_reduced_capability"] is True
        assert benign_res["final_classification"] == "INCONCLUSIVE"
        assert any("cannot confirm legitimacy" in r.lower() for r in benign_res["reasons"])


@pytest.fixture(scope="module")
def trained_model_path():
    """Trains an actual model on data/sample_dataset.csv into a temporary file."""
    data_path = Config.BASE_DIR / "data" / "sample_dataset.csv"
    temp_dir = tempfile.mkdtemp()
    output_model = Path(temp_dir) / "test_model.joblib"

    metrics = train_and_evaluate(
        data_path=data_path,
        output_path=output_model,
        test_size=0.25,
        random_state=42
    )

    assert output_model.exists()
    assert metrics["accuracy"] >= 0.70
    assert metrics["f1_score"] >= 0.70

    return output_model


class TestModelTrainingAndInference:
    """Tests actual training pipeline execution and inference evaluation."""

    def test_trained_model_inference_legitimate(self, trained_model_path):
        detector = ModelDetector(model_path=trained_model_path)
        assert detector.is_loaded is True

        res = detector.predict(
            subject="Sprint retrospective meeting notes",
            body="Hi team, attached are the action items from our weekly engineering sync. See you at tomorrow's standup."
        )

        assert res["is_available"] is True
        assert res["prediction"] == "LEGITIMATE"
        assert res["phishing_probability"] < 0.50
        assert res["legitimate_probability"] > 0.50
        assert res["confidence"] > 0.50

    def test_trained_model_inference_phishing(self, trained_model_path):
        detector = ModelDetector(model_path=trained_model_path)
        assert detector.is_loaded is True

        res = detector.predict(
            subject="URGENT: Unauthorized login attempt - verify account",
            body="Your account has been suspended due to security risks. Reset your password immediately and update payment information."
        )

        assert res["is_available"] is True
        assert res["prediction"] == "PHISHING"
        assert res["phishing_probability"] >= 0.50
        assert len(res["contributing_features"]) > 0

    def test_edge_case_challenging_false_positive_analysis(self, trained_model_path):
        """Demonstrates investigation of borderline cases.
        
        Example: An urgent corporate security alert sent by legitimate IT containing words
        like 'security notice' and 'update password', which could trigger high phishing probability.
        """
        detector = ModelDetector(model_path=trained_model_path)
        combined = CombinedDetector(model_detector=detector)

        legit_security_advisory = (
            "Internal IT Security Bulletin: Scheduled quarterly password rotation reminder. "
            "Please visit the internal intranet portal to update your corporate SSO password."
        )

        res = combined.analyze(subject="IT Security Advisory: Password Rotation", body=legit_security_advisory)
        # Verify that the system provides clear rationale and doesn't crash on nuanced emails
        assert "reasons" in res
        assert res["risk_score"] >= 0.0
