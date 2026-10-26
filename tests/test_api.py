"""Automated tests for PhishGuard Flask Application and REST API endpoints."""

import json
import pytest
from app import create_app
from config import TestingConfig


@pytest.fixture
def client():
    """Configures a test client for Flask API testing."""
    test_app = create_app(TestingConfig)
    with test_app.test_client() as test_client:
        yield test_client


class TestAPIEndpoints:
    """Test suite for PhishGuard API endpoints and validation."""

    def test_get_index_page(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert b"PhishGuard" in response.data
        assert b"Think Before You Click" in response.data

    def test_health_check_endpoint(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "healthy"
        assert "components" in data
        assert "dfa_engine" in data["components"]
        assert "machine_learning_model" in data["components"]

    def test_analyze_valid_payload(self, client):
        payload = {
            "subject": "Urgent: Verify your account immediately",
            "body": "Your bank profile has been suspended. Confirm your password at http://192.168.1.1/auth.",
            "sender": "alert@fake-bank.xyz"
        }
        response = client.post(
            "/api/analyze",
            data=json.dumps(payload),
            content_type="application/json"
        )
        assert response.status_code == 200
        data = response.get_json()
        assert "final_classification" in data
        assert "risk_score" in data
        assert "dfa" in data
        assert "machine_learning" in data
        assert "slm" in data
        assert "reasons" in data
        assert "disclaimer" in data
        assert data["risk_score"] > 50.0

    def test_analyze_empty_subject_and_body_rejected(self, client):
        payload = {
            "subject": "   ",
            "body": "",
            "sender": "someone@example.com"
        }
        response = client.post(
            "/api/analyze",
            data=json.dumps(payload),
            content_type="application/json"
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data

    def test_analyze_non_json_content_type(self, client):
        response = client.post(
            "/api/analyze",
            data="raw text string",
            content_type="text/plain"
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "Invalid Content-Type" in data["error"]

    def test_analyze_malformed_json_body(self, client):
        response = client.post(
            "/api/analyze",
            data='{"subject": "test", broken json',
            content_type="application/json"
        )
        assert response.status_code == 400

    def test_analyze_invalid_field_types(self, client):
        payload = {
            "subject": 12345,  # Invalid: integer instead of string
            "body": "Normal body text"
        }
        response = client.post(
            "/api/analyze",
            data=json.dumps(payload),
            content_type="application/json"
        )
        assert response.status_code == 400
        data = response.get_json()
        assert "Invalid field types" in data["error"]

    def test_analyze_oversized_subject_rejected(self, client):
        oversized_subject = "A" * 600  # Max is 500
        payload = {
            "subject": oversized_subject,
            "body": "Regular body"
        }
        response = client.post(
            "/api/analyze",
            data=json.dumps(payload),
            content_type="application/json"
        )
        assert response.status_code == 413
        data = response.get_json()
        assert "exceeds maximum allowed length" in data["error"]

    def test_invalid_http_method(self, client):
        response = client.get("/api/analyze")
        assert response.status_code == 405

    def test_api_not_found(self, client):
        response = client.get("/api/nonexistent_endpoint")
        assert response.status_code == 404
        data = response.get_json()
        assert "API endpoint not found" in data["error"]
