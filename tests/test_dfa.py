"""Automated tests for the PhishGuard DFA Detection Module."""

import pytest
from dfa_detector import DFADetector, PhraseDFA, URLStructureDFA


class TestPhraseDFA:
    """Test suite for Token-based Phrase DFA."""

    def setup_method(self):
        self.dfa = PhraseDFA()
        self.dfa.add_pattern("verify your account", "Credential Harvesting", "HIGH", "Verification request")
        self.dfa.add_pattern("account suspended", "Account Threat", "HIGH", "Suspension threat")
        self.dfa.add_pattern("urgent action required", "Urgency", "MEDIUM", "Urgency cue")
        self.dfa.compile()

    def test_exact_phrase_match(self):
        text = "Please verify your account immediately."
        matches, transitions = self.dfa.scan(text)
        assert len(matches) == 1
        assert matches[0]["pattern"] == "verify your account"
        assert matches[0]["category"] == "Credential Harvesting"
        assert transitions > 0

    def test_case_insensitivity_and_punctuation(self):
        text = "URGENT ACTION REQUIRED: Your ACCOUNT SUSPENDED!!!"
        matches, _ = self.dfa.scan(text)
        patterns = {m["pattern"] for m in matches}
        assert "urgent action required" in patterns
        assert "account suspended" in patterns

    def test_no_false_match_on_partial_tokens(self):
        # "accounting" should not trigger "account" if token boundary matters
        text = "We are reviewing our corporate accounting procedures."
        matches, _ = self.dfa.scan(text)
        assert len(matches) == 0

    def test_empty_string(self):
        matches, transitions = self.dfa.scan("")
        assert matches == []
        assert transitions == 0


class TestURLStructureDFA:
    """Test suite for character-level URL Structure DFA."""

    def setup_method(self):
        self.url_dfa = URLStructureDFA()

    def test_detect_ipv4_host(self):
        res = self.url_dfa.analyze_url("http://192.168.1.50/login.php")
        anomaly_types = [a["type"] for a in res["anomalies"]]
        assert "IP_ADDRESS_HOST" in anomaly_types
        assert res["transitions_executed"] > 0

    def test_detect_userinfo_spoofing(self):
        res = self.url_dfa.analyze_url("https://secure.chase.com@phish-domain.com/auth")
        anomaly_types = [a["type"] for a in res["anomalies"]]
        assert "USERINFO_SPOOFING" in anomaly_types

    def test_detect_suspicious_tld(self):
        res = self.url_dfa.analyze_url("http://account-update.xyz/verify")
        anomaly_types = [a["type"] for a in res["anomalies"]]
        assert "SUSPICIOUS_TLD" in anomaly_types

    def test_detect_unusual_port(self):
        res = self.url_dfa.analyze_url("http://portal.example.com:8443/login")
        anomaly_types = [a["type"] for a in res["anomalies"]]
        assert "UNUSUAL_PORT" in anomaly_types

    def test_legitimate_url_clean(self):
        res = self.url_dfa.analyze_url("https://www.google.com/search?q=cybersecurity")
        assert len(res["anomalies"]) == 0


class TestDFADetectorIntegration:
    """Integrated tests for master DFADetector."""

    def setup_method(self):
        self.detector = DFADetector()

    def test_phishing_email_detection(self):
        subject = "Urgent: Account Suspended"
        body = "We detected suspicious activity. Please verify your account at http://192.168.0.1/verify now."
        result = self.detector.analyze(subject, body)

        assert result["dfa_risk_score"] > 50.0
        assert result["indicators_count"] >= 3
        phrase_patterns = [p["pattern"] for p in result["detected_phrases"]]
        assert "account suspended" in phrase_patterns
        assert "verify your account" in phrase_patterns

        assert len(result["url_findings"]) == 1
        assert any(a["type"] == "IP_ADDRESS_HOST" for a in result["url_findings"][0]["anomalies"])

    def test_legitimate_email_clean(self):
        subject = "Team sync meeting notes - Sprint 42"
        body = "Hi team, please find attached the minutes from our Monday engineering standup. See you all tomorrow at 10am."
        result = self.detector.analyze(subject, body)

        assert result["dfa_risk_score"] == 0.0
        assert len(result["detected_phrases"]) == 0
        assert len(result["url_findings"]) == 0
        assert result["indicators_count"] == 0

    def test_empty_input(self):
        result = self.detector.analyze("", "")
        assert result["dfa_risk_score"] == 0.0
        assert result["indicators_count"] == 0
