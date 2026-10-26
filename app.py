"""PhishGuard Flask Application and REST API.

Provides endpoints for email phishing analysis, health checks, and dashboard rendering.
Strictly validates user inputs, enforces payload size limits, and returns structured JSON responses.
"""

from pathlib import Path
import logging
from typing import Dict, Any

from flask import Flask, request, jsonify, render_template

from config import Config
from combined_detector import CombinedDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phishguard")


def create_app(config_class=Config) -> Flask:
    """Application factory for PhishGuard."""
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static"
    )
    if isinstance(config_class, type):
        cfg = config_class()
    else:
        cfg = config_class

    app.config.from_object(cfg)

    # Initialize combined detection engine
    detector = CombinedDetector(config=cfg)

    @app.route("/")
    def index():
        """Serves the main PhishGuard frontend dashboard."""
        return render_template("index.html")

    @app.route("/api/health", methods=["GET"])
    def health_check():
        """Returns backend service status and component availability."""
        # Refresh model status dynamically if a model was placed on disk
        if not detector.model.is_loaded:
            detector.model.load_model()

        return jsonify({
            "status": "healthy",
            "service": "PhishGuard",
            "version": "1.0.0",
            "components": {
                "dfa_engine": {
                    "status": "active",
                    "registered_phrases": len(detector.dfa.phrase_dfa.states)
                },
                "machine_learning_model": {
                    "is_loaded": detector.model.is_loaded,
                    "model_path": str(detector.model.model_path),
                    "decision_threshold": detector.model.threshold,
                    "status_message": detector.model.load_error or "Loaded and operational."
                },
                "slm_interface": {
                    "is_available": detector.slm.is_available,
                    "model_name": detector.slm.model_name,
                    "enabled": detector.slm.enabled
                }
            }
        }), 200

    @app.route("/api/analyze", methods=["POST"])
    def analyze_email():
        """Analyzes email content for phishing indicators.
        
        Supports both:
          1. {"email_text": "..."}
          2. {"subject": "...", "body": "...", "sender": "..."}
        """
        # Validate Content-Type
        if not request.is_json:
            return jsonify({
                "error": "Invalid Content-Type. Request body must be valid JSON with 'application/json' header."
            }), 400

        try:
            data = request.get_json(silent=True)
            if data is None:
                return jsonify({"error": "Malformed or missing JSON payload in request body."}), 400
        except Exception:
            return jsonify({"error": "Failed to parse JSON body."}), 400

        max_body_len = app.config.get("MAX_EMAIL_BODY_LENGTH", 50000)
        max_subject_len = app.config.get("MAX_EMAIL_SUBJECT_LENGTH", 500)
        max_sender_len = app.config.get("MAX_EMAIL_SENDER_LENGTH", 320)

        # Check for unified "email_text" field
        if "email_text" in data:
            email_text = data.get("email_text")
            if not isinstance(email_text, str):
                return jsonify({
                    "error": "Invalid field type. 'email_text' must be a string."
                }), 400

            if len(email_text) > max_body_len:
                return jsonify({
                    "error": f"Email text exceeds maximum allowed length of {max_body_len} characters."
                }), 413

            clean_text = email_text.strip()
            if not clean_text:
                return jsonify({
                    "error": "Cannot analyze an empty email. Please provide non-whitespace email text."
                }), 400

            clean_subject = ""
            clean_body = clean_text
            clean_sender = ""

        else:
            # Multi-field format: subject, body, sender
            subject = data.get("subject", "")
            body = data.get("body", "")
            sender = data.get("sender", "")

            if not isinstance(subject, str) or not isinstance(body, str) or not isinstance(sender, str):
                return jsonify({
                    "error": "Invalid field types. 'subject', 'body', and 'sender' must be strings."
                }), 400

            if len(subject) > max_subject_len:
                return jsonify({
                    "error": f"Email subject exceeds maximum allowed length of {max_subject_len} characters."
                }), 413

            if len(body) > max_body_len:
                return jsonify({
                    "error": f"Email body exceeds maximum allowed length of {max_body_len} characters."
                }), 413

            if len(sender) > max_sender_len:
                return jsonify({
                    "error": f"Sender field exceeds maximum allowed length of {max_sender_len} characters."
                }), 413

            clean_subject = subject.strip()
            clean_body = body.strip()
            clean_sender = sender.strip()

            if not clean_subject and not clean_body:
                return jsonify({
                    "error": "Cannot analyze an empty email. Please provide an email subject or body text."
                }), 400

        # Ensure model availability is checked
        if not detector.model.is_loaded:
            detector.model.load_model()

        # Run multi-modal detection
        try:
            analysis_result = detector.analyze(
                subject=clean_subject,
                body=clean_body,
                sender=clean_sender
            )
            return jsonify(analysis_result), 200
        except Exception as exc:
            logger.exception("Unexpected error during email analysis: %s", exc)
            return jsonify({
                "error": "An internal server error occurred while analyzing the email."
            }), 500

    # Error Handlers
    @app.errorhandler(400)
    def handle_bad_request(err):
        return jsonify({"error": "Bad request", "message": str(err)}), 400

    @app.errorhandler(404)
    def handle_not_found(err):
        if request.path.startswith("/api/"):
            return jsonify({"error": "API endpoint not found", "path": request.path}), 404
        return render_template("index.html"), 404

    @app.errorhandler(405)
    def handle_method_not_allowed(err):
        return jsonify({"error": "HTTP method not allowed for this endpoint."}), 405

    @app.errorhandler(413)
    def handle_payload_too_large(err):
        return jsonify({"error": "Payload too large. Exceeds server request limits."}), 413

    @app.errorhandler(500)
    def handle_server_error(err):
        logger.exception("Internal 500 error encountered: %s", err)
        return jsonify({"error": "Internal server error."}), 500

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
