"""PhishGuard Configuration Module.

Central configuration settings for security parameters, DFA pattern weights,
machine learning paths, thresholds, and optional SLM integration settings.
"""

from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent

class Config:
    """Application configuration parameters."""
    
    BASE_DIR = BASE_DIR
    
    # Flask settings
    SECRET_KEY = os.environ.get("PHISHGUARD_SECRET_KEY", "dev-phishguard-key-change-in-prod")
    DEBUG = False
    TESTING = False
    
    # Input validation constraints
    MAX_EMAIL_SUBJECT_LENGTH = 500
    MAX_EMAIL_BODY_LENGTH = 50000
    MAX_EMAIL_SENDER_LENGTH = 320
    MAX_CONTENT_LENGTH = 2 * 1024 * 1024  # 2MB HTTP payload limit
    
    # Model configuration
    MODEL_DIR = BASE_DIR / "models"
    MODEL_PATH = MODEL_DIR / "phishing_model.joblib"
    MODEL_DECISION_THRESHOLD = 0.50
    
    # Combined Risk Scoring Thresholds (0 - 100 scale)
    # < 30: LIKELY_LEGITIMATE
    # 30 - 65: SUSPICIOUS
    # > 65: PHISHING
    THRESHOLD_LEGITIMATE = 30.0
    THRESHOLD_SUSPICIOUS = 65.0
    
    # Risk weights when all components are active:
    # If ML model is unavailable, the system transparently shifts to
    # Reduced Capability Mode using rule-based and DFA structural weights.
    WEIGHT_ML_MODEL = 0.50
    WEIGHT_DFA = 0.35
    WEIGHT_STRUCTURAL_URL = 0.15
    
    # DFA Severity levels and numerical weights
    DFA_SEVERITY_WEIGHTS = {
        "CRITICAL": 35.0,
        "HIGH": 25.0,
        "MEDIUM": 15.0,
        "LOW": 8.0,
    }
    
    # SLM (Small Language Model) Configuration (Phase-ready modular interface)
    SLM_ENABLED = os.environ.get("PHISHGUARD_SLM_ENABLED", "false").lower() == "true"
    SLM_MODEL_NAME = os.environ.get("PHISHGUARD_SLM_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
    SLM_ENDPOINT = os.environ.get("PHISHGUARD_SLM_ENDPOINT", "")
    SLM_MAX_NEW_TOKENS = 256
    SLM_TEMPERATURE = 0.1

class TestingConfig(Config):
    """Testing specific configuration overrides."""
    TESTING = True
    DEBUG = True
    MODEL_PATH = BASE_DIR / "models" / "test_model.joblib"
