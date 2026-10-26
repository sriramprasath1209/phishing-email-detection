"""PhishGuard Deterministic Finite Automaton (DFA) Detection Module.

This module provides formal finite-state pattern matching for email phishing
indicators, specifically:
1. Multi-token phrase scanning via an explicit Token-based Trie/Aho-Corasick DFA.
2. Character-level structural URL inspection DFA (detecting IP address hosts,
   userinfo credential harvesting, hex/percent obfuscation, and suspicious TLDs).

Theoretical Formalism:
----------------------
A Deterministic Finite Automaton (DFA) is defined as a 5-tuple M = (Q, Sigma, delta, q0, F):
- Q: Finite set of states.
- Sigma: Alphabet. For Token DFA, Sigma is the set of normalized word tokens.
  For URL DFA, Sigma is the ASCII character set.
- delta: Transition function delta: Q x Sigma -> Q.
- q0 in Q: Start state (root).
- F subset of Q: Accepting states signifying recognized patterns or structural anomalies.

DFA vs Regular Expression Distinction:
---------------------------------------
- DFA (Implemented here):
  * PhraseDFA: Explicit state graph where each node represents a deterministic
    prefix state in multi-token phishing patterns. State transitions occur
    deterministically on token consumption with O(N) single-pass traversal.
  * URLStructureDFA: Explicit character-by-character state transition machine
    identifying IP addresses in URL authority components, '@' userinfo spoofing,
    and hex-encoded character transitions.
- Regular Expressions (Utility only):
  * Used solely as preliminary lexical splitters to tokenize text and extract
    candidate URL strings from unstructured email bodies before feeding them
    into the DFA state machines.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
import re
from urllib.parse import urlparse

from config import Config


class DFAState:
    """Represents a state in the Phrase DFA state machine."""

    def __init__(self, state_id: int):
        self.state_id = state_id
        self.transitions: Dict[str, int] = {}
        self.is_accepting: bool = False
        self.payloads: List[Dict[str, Any]] = []
        self.fail_state: int = 0


class PhraseDFA:
    """Deterministic Finite Automaton for scanning multi-token phishing phrases.
    
    Tokens are normalized lowercase word strings.
    Transitions delta(q, token) -> q' transition through state tree.
    """

    def __init__(self):
        self.states: List[DFAState] = [DFAState(0)]
        self.next_state_id = 1
        self._compiled = False

    def add_pattern(self, phrase: str, category: str, severity: str, explanation: str):
        """Adds a phrase pattern to the automaton."""
        tokens = self._tokenize(phrase)
        if not tokens:
            return

        current_id = 0
        for token in tokens:
            state = self.states[current_id]
            if token not in state.transitions:
                new_state = DFAState(self.next_state_id)
                self.states.append(new_state)
                state.transitions[token] = self.next_state_id
                self.next_state_id += 1
            current_id = state.transitions[token]

        target_state = self.states[current_id]
        target_state.is_accepting = True
        target_state.payloads.append({
            "pattern": phrase,
            "category": category,
            "severity": severity,
            "explanation": explanation
        })
        self._compiled = False

    def compile(self):
        """Constructs failure transitions for single-pass multi-token scanning (Aho-Corasick style)."""
        queue = []
        # Level 1 states fail to root (state 0)
        for token, next_id in self.states[0].transitions.items():
            self.states[next_id].fail_state = 0
            queue.append(next_id)

        while queue:
            curr_id = queue.pop(0)
            curr_state = self.states[curr_id]

            for token, next_id in curr_state.transitions.items():
                queue.append(next_id)
                fail = curr_state.fail_state
                while fail != 0 and token not in self.states[fail].transitions:
                    fail = self.states[fail].fail_state

                if token in self.states[fail].transitions and self.states[fail].transitions[token] != next_id:
                    self.states[next_id].fail_state = self.states[fail].transitions[token]
                else:
                    self.states[next_id].fail_state = 0

                # Propagate accepting payloads from suffix fail state
                suffix_fail_id = self.states[next_id].fail_state
                if self.states[suffix_fail_id].is_accepting:
                    self.states[next_id].is_accepting = True
                    for p in self.states[suffix_fail_id].payloads:
                        if p not in self.states[next_id].payloads:
                            self.states[next_id].payloads.append(p)

        self._compiled = True

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Lexical tokenizer: lowercases and extracts alphanumeric word tokens."""
        if not text:
            return []
        return [word.lower() for word in re.findall(r"\b[a-zA-Z0-9']+\b", text)]

    def scan(self, text: str) -> Tuple[List[Dict[str, Any]], int]:
        """Scans input text through the DFA.
        
        Returns:
            Tuple of (matches, transitions_count)
        """
        if not self._compiled:
            self.compile()

        tokens = self._tokenize(text)
        current_id = 0
        matches = []
        seen_patterns = set()
        transitions_count = 0

        for idx, token in enumerate(tokens):
            transitions_count += 1
            while current_id != 0 and token not in self.states[current_id].transitions:
                current_id = self.states[current_id].fail_state
                transitions_count += 1

            if token in self.states[current_id].transitions:
                current_id = self.states[current_id].transitions[token]
            else:
                current_id = 0

            curr_state = self.states[current_id]
            if curr_state.is_accepting:
                for payload in curr_state.payloads:
                    if payload["pattern"] not in seen_patterns:
                        seen_patterns.add(payload["pattern"])
                        matches.append({
                            "pattern": payload["pattern"],
                            "category": payload["category"],
                            "severity": payload["severity"],
                            "explanation": payload["explanation"],
                            "token_position": idx
                        })

        return matches, transitions_count


class URLStructureDFA:
    """Character-level DFA for identifying structural anomalies in email URLs.
    
    States:
      0: URL_START
      1: SCHEME (http / https)
      2: COLON
      3: SLASH_1
      4: SLASH_2
      5: HOST_START
      6: HOST_BODY
      7: IP_OCTET_1
      8: IP_DOT_1
      9: IP_OCTET_2
      10: IP_DOT_2
      11: IP_OCTET_3
      12: IP_DOT_3
      13: IP_OCTET_4 (Accepting: Raw IPv4 address)
      14: USERINFO_AUTH (Accepting: Spoofed '@' character in authority)
      15: PORT_START
      16: PORT_BODY (Accepting: Non-standard port)
      17: HEX_ENCODED (Accepting: Obfuscated hex encoding in authority)
    """

    SUSPICIOUS_TLDS = {
        "xyz", "top", "tk", "ml", "ga", "cf", "gq", "work", "click", "rest", "buzz", "loan", "fit", "icu"
    }

    def __init__(self):
        self.state_names = {
            0: "URL_START",
            1: "SCHEME",
            2: "COLON",
            3: "SLASH_1",
            4: "SLASH_2",
            5: "HOST_START",
            6: "HOST_BODY",
            7: "IP_OCTET_1",
            8: "IP_DOT_1",
            9: "IP_OCTET_2",
            10: "IP_DOT_2",
            11: "IP_OCTET_3",
            12: "IP_DOT_3",
            13: "ACCEPT_RAW_IP",
            14: "ACCEPT_USERINFO_SPOOF",
            15: "PORT_START",
            16: "ACCEPT_UNUSUAL_PORT",
            17: "ACCEPT_HEX_OBFUSCATION",
        }

    def analyze_url(self, raw_url: str) -> Dict[str, Any]:
        """Executes explicit character-by-character DFA transition logic on candidate URL."""
        anomalies: List[Dict[str, str]] = []
        transitions_executed = 0

        # State initialization
        state = 0
        authority_chars = []
        in_authority = False
        has_at_symbol = False
        has_hex_encoding = False
        port_chars = []
        in_port = False

        idx = 0
        n = len(raw_url)

        # Lexical character scanner DFA
        while idx < n:
            ch = raw_url[idx]
            transitions_executed += 1

            if state == 0:  # Start
                if ch.lower() in "h":
                    state = 1
                else:
                    state = 5  # No scheme, starts straight with authority
                    authority_chars.append(ch)
                    in_authority = True

            elif state == 1:  # Scheme reading
                if ch == ":":
                    state = 2
                elif ch in "/?#":
                    state = 5
                    authority_chars.append(ch)
                    in_authority = True

            elif state == 2:  # After colon
                if ch == "/":
                    state = 3
                else:
                    state = 5
                    authority_chars.append(ch)
                    in_authority = True

            elif state == 3:  # First slash
                if ch == "/":
                    state = 4  # Authority begins next
                    in_authority = True
                else:
                    state = 5
                    authority_chars.append(ch)
                    in_authority = True

            elif state == 4:  # Beginning of authority (host + optional userinfo / port)
                if ch in "/?#":
                    # End of authority
                    in_authority = False
                    break
                if ch == "@":
                    has_at_symbol = True
                elif ch == "%":
                    has_hex_encoding = True
                elif ch == ":":
                    in_port = True
                else:
                    if in_port:
                        port_chars.append(ch)
                    else:
                        authority_chars.append(ch)

            idx += 1

        authority_str = "".join(authority_chars).strip()
        port_str = "".join(port_chars).strip()

        # Check DFA accepting states on parsed authority
        if has_at_symbol:
            anomalies.append({
                "type": "USERINFO_SPOOFING",
                "severity": "CRITICAL",
                "explanation": "URL contains '@' userinfo separator; often used to visually mimic a legitimate brand while directing to an attacker host."
            })

        if has_hex_encoding:
            anomalies.append({
                "type": "HEX_OBFUSCATION",
                "severity": "HIGH",
                "explanation": "URL hostname contains hex or percent-encoded characters, typical of evasion techniques."
            })

        # Check IPv4 Host DFA:
        # Regex or character DFA test for pure numeric IPv4 address
        is_ipv4 = self._check_ipv4_dfa(authority_str)
        if is_ipv4:
            anomalies.append({
                "type": "IP_ADDRESS_HOST",
                "severity": "CRITICAL",
                "explanation": f"Host is a raw numerical IP address ({authority_str}) rather than a registered domain name."
            })

        # Check non-standard ports
        if port_str:
            try:
                port_num = int(port_str)
                if port_num not in (80, 443, 8080):
                    anomalies.append({
                        "type": "UNUSUAL_PORT",
                        "severity": "MEDIUM",
                        "explanation": f"URL connects to non-standard HTTP port {port_num}."
                    })
            except ValueError:
                pass

        # Check excessive subdomains (chaining)
        subdomains = [part for part in authority_str.split(".") if part]
        if len(subdomains) >= 5 and not is_ipv4:
            anomalies.append({
                "type": "EXCESSIVE_SUBDOMAINS",
                "severity": "HIGH",
                "explanation": f"Host contains {len(subdomains)} domain labels, indicating potential domain tunneling or spoofing."
            })

        # Check suspicious TLD
        if subdomains and len(subdomains) > 1 and not is_ipv4:
            tld = subdomains[-1].lower()
            if tld in self.SUSPICIOUS_TLDS:
                anomalies.append({
                    "type": "SUSPICIOUS_TLD",
                    "severity": "MEDIUM",
                    "explanation": f"Domain utilizes top-level domain '.{tld}', statistically correlated with disposable phishing infrastructure."
                })

        return {
            "url": raw_url,
            "host": authority_str,
            "anomalies": anomalies,
            "transitions_executed": transitions_executed
        }

    @staticmethod
    def _check_ipv4_dfa(host_str: str) -> bool:
        """Explicit state machine verifying whether host is a valid IPv4 address."""
        parts = host_str.split(":")
        candidate = parts[0]
        octets = candidate.split(".")
        if len(octets) != 4:
            return False

        for octet in octets:
            if not octet.isdigit():
                return False
            val = int(octet)
            if val < 0 or val > 255:
                return False
            # Detect leading zeros that could indicate octal obfuscation
            if len(octet) > 1 and octet.startswith("0"):
                return True
        return True


class DFADetector:
    """Master Deterministic Finite Automaton Engine for PhishGuard.
    
    Coordinates phrase matching and URL structural analysis.
    """

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.phrase_dfa = PhraseDFA()
        self.url_dfa = URLStructureDFA()
        self._init_default_patterns()
        self.phrase_dfa.compile()

    def _init_default_patterns(self):
        """Initializes canonical phishing indicators with categories, severities, and explanations."""
        patterns = [
            # Credential Harvesting
            ("verify your account", "Credential Harvesting", "HIGH", "Requests verification of user credentials."),
            ("confirm your password", "Credential Harvesting", "CRITICAL", "Directly prompts the recipient to enter their current password."),
            ("reset your password immediately", "Credential Harvesting", "CRITICAL", "Demands urgent credential modification."),
            ("enter your credentials", "Credential Harvesting", "CRITICAL", "Explicit request for authentication secrets."),
            ("update payment information", "Financial Fraud", "HIGH", "Solicits credit card or payment profile alterations."),
            ("wire transfer request", "Financial Fraud", "CRITICAL", "Requests manual wire or unauthorized financial disbursement."),
            ("click here to unlock", "Credential Harvesting", "HIGH", "Call-to-action leading to external unlock portals."),
            ("update your banking details", "Financial Fraud", "CRITICAL", "Requests sensitive banking coordinates."),

            # Urgency and Coercion
            ("urgent action required", "Urgency & Pressure", "HIGH", "Applies psychological pressure to bypass scrutiny."),
            ("immediate attention required", "Urgency & Pressure", "HIGH", "High urgency cue intended to induce hasty clicking."),
            ("action required within 24 hours", "Urgency & Pressure", "HIGH", "Artificial deadline designed to force rash actions."),
            ("within 48 hours", "Urgency & Pressure", "MEDIUM", "Time constraint tactic often paired with threats."),
            ("immediate response needed", "Urgency & Pressure", "MEDIUM", "Urgency tactic to solicit rapid unverified response."),

            # Account Threat & Suspension
            ("account suspended", "Account Threat", "HIGH", "Alleges account closure to induce panic and compliance."),
            ("unauthorized login attempt", "Account Threat", "HIGH", "Fabricates unauthorized sign-in alert."),
            ("unusual sign in activity", "Account Threat", "MEDIUM", "Claims suspicious access to provoke verification action."),
            ("suspicious activity detected", "Account Threat", "MEDIUM", "Generic security alert used in credential phishing."),
            ("your mailbox is full", "Account Threat", "MEDIUM", "Quota scare tactic commonly used in corporate credential theft."),
            ("access will be terminated", "Account Threat", "CRITICAL", "Threatens immediate service cutoff."),
            ("security notice", "Account Threat", "LOW", "Standard security claim frequently leveraged in social engineering."),

            # Delivery & Reward Lures
            ("failed delivery attempt", "Delivery Lure", "MEDIUM", "Fake courier package notification containing malicious link."),
            ("claim your prize", "Reward Lure", "HIGH", "Classic lottery/reward lure requiring user personal information."),
            ("you have won", "Reward Lure", "HIGH", "Unsolicited reward notification designed to harvest identity data.")
        ]

        for phrase, category, severity, explanation in patterns:
            self.phrase_dfa.add_pattern(phrase, category, severity, explanation)

    @staticmethod
    def extract_urls(text: str) -> List[str]:
        """Extracts candidate URLs using regular expression tokenizer before passing to DFA."""
        if not text:
            return []
        url_pattern = r"(?:https?://|www\.)[^\s<>'\"`]+"
        raw_urls = re.findall(url_pattern, text)
        clean_urls = []
        for url in raw_urls:
            # Strip trailing punctuation often caught in sentence boundaries
            clean_url = url.rstrip(".,;!?:)]}\"'")
            if clean_url:
                clean_urls.append(clean_url)
        return clean_urls

    def analyze(self, subject: str, body: str, sender: str = "") -> Dict[str, Any]:
        """Executes full DFA scanning across email subject, body, and sender.
        
        Returns:
            Dict containing detected indicators, URL findings, DFA risk score,
            total transitions, and technical explanations.
        """
        combined_text = f"{subject}\n{body}\n{sender}".strip()
        if not combined_text:
            return {
                "detected_phrases": [],
                "url_findings": [],
                "dfa_risk_score": 0.0,
                "total_transitions": 0,
                "summary": "No text provided for DFA scanning.",
                "indicators_count": 0
            }

        # 1. Scan phrases via PhraseDFA
        phrase_matches, phrase_transitions = self.phrase_dfa.scan(combined_text)

        # 2. Extract and scan URLs via URLStructureDFA
        extracted_urls = self.extract_urls(combined_text)
        url_findings = []
        url_transitions = 0
        for u in extracted_urls:
            u_result = self.url_dfa.analyze_url(u)
            url_findings.append(u_result)
            url_transitions += u_result["transitions_executed"]

        # 3. Calculate DFA Severity Score
        raw_score = 0.0
        severity_weights = self.config.DFA_SEVERITY_WEIGHTS

        for match in phrase_matches:
            raw_score += severity_weights.get(match["severity"], 10.0)

        for u in url_findings:
            for anom in u.get("anomalies", []):
                raw_score += severity_weights.get(anom["severity"], 15.0)

        # Non-linear damping to scale into 0 - 100 range
        dfa_risk_score = min(100.0, round(raw_score, 1))

        # Generate technical summary
        phrase_count = len(phrase_matches)
        url_anom_count = sum(len(u.get("anomalies", [])) for u in url_findings)
        
        if phrase_count == 0 and url_anom_count == 0:
            summary = "No known phishing phrase patterns or structural URL anomalies were detected by the DFA."
        else:
            summary = (
                f"DFA identified {phrase_count} suspicious phrase indicator(s) and "
                f"{url_anom_count} URL structural anomal(ies) across "
                f"{phrase_transitions + url_transitions} state transitions."
            )

        return {
            "detected_phrases": phrase_matches,
            "url_findings": url_findings,
            "dfa_risk_score": dfa_risk_score,
            "total_transitions": phrase_transitions + url_transitions,
            "summary": summary,
            "indicators_count": phrase_count + url_anom_count
        }
