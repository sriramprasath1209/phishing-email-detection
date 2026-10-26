/**
 * PhishGuard Cybersecurity Dashboard Frontend Script
 *
 * Implements client-side input validation, health polling, preset loaders,
 * and safe DOM rendering using textContent to prevent XSS.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const form = document.getElementById("emailAnalysisForm");
    const subjectInput = document.getElementById("emailSubject");
    const bodyInput = document.getElementById("emailBody");
    const senderInput = document.getElementById("emailSender");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const analyzeBtnText = document.getElementById("analyzeBtnText");
    const analyzeSpinner = document.getElementById("analyzeSpinner");
    const clearBtn = document.getElementById("clearBtn");
    const charCountEl = document.getElementById("charCount");
    const formAlert = document.getElementById("formAlert");
    const formAlertText = document.getElementById("formAlertText");

    // Header Status Elements
    const backendDot = document.getElementById("backendDot");
    const backendStatusText = document.getElementById("backendStatusText");
    const modelDot = document.getElementById("modelDot");
    const modelStatusText = document.getElementById("modelStatusText");

    // Results State Elements
    const emptyState = document.getElementById("emptyState");
    const loadingState = document.getElementById("loadingState");
    const populatedResults = document.getElementById("populatedResults");

    // Results Summary Elements
    const verdictBanner = document.getElementById("verdictBanner");
    const riskLevelBadge = document.getElementById("riskLevelBadge");
    const finalClassification = document.getElementById("finalClassification");
    const operatingModeTag = document.getElementById("operatingModeTag");
    const riskScoreValue = document.getElementById("riskScoreValue");
    const scoreCircle = document.getElementById("scoreCircle");

    // Metric Elements
    const dfaScoreBadge = document.getElementById("dfaScoreBadge");
    const dfaTransitionsCount = document.getElementById("dfaTransitionsCount");
    const dfaPhrasesCount = document.getElementById("dfaPhrasesCount");
    const dfaUrlAnomaliesCount = document.getElementById("dfaUrlAnomaliesCount");
    const dfaSummaryText = document.getElementById("dfaSummaryText");

    const mlStatusBadge = document.getElementById("mlStatusBadge");
    const mlProbabilityValue = document.getElementById("mlProbabilityValue");
    const mlConfidenceValue = document.getElementById("mlConfidenceValue");
    const mlThresholdValue = document.getElementById("mlThresholdValue");
    const mlStatusMessage = document.getElementById("mlStatusMessage");

    const slmBadge = document.getElementById("slmBadge");
    const slmStatusText = document.getElementById("slmStatusText");

    // Detail Containers
    const phrasesGroup = document.getElementById("phrasesGroup");
    const phrasesContainer = document.getElementById("phrasesContainer");
    const urlsGroup = document.getElementById("urlsGroup");
    const urlsContainer = document.getElementById("urlsContainer");
    const mlFeaturesGroup = document.getElementById("mlFeaturesGroup");
    const featuresContainer = document.getElementById("featuresContainer");
    const reasonsList = document.getElementById("reasonsList");
    const disclaimerText = document.getElementById("disclaimerText");

    // Preset Data
    const PRESETS = {
        legit: {
            sender: "alex.turner@internal-engineering.com",
            subject: "Sprint 42 Retrospective & Q3 Architecture Planning",
            body: "Hi team,\n\nOur sprint retrospective will take place tomorrow at 10:00 AM PST in Conference Room B. Please add your discussion topics and pull request links to the shared wiki document before end of day.\n\nLooking forward to reviewing the database migration benchmarks!\n\nBest,\nAlex"
        },
        bank: {
            sender: "service-alert@chase-support-sec.xyz",
            subject: "URGENT: Your account has been suspended",
            body: "Security Notification:\n\nWe detected an unauthorized login attempt from an unrecognized IP address (185.220.101.5). For your protection, your banking profile has been temporarily suspended.\n\nImmediate attention required: You must confirm your password and verify your account within 24 hours at http://192.168.1.50/verify to unlock access.\n\nFailure to confirm will result in permanent closure."
        },
        courier: {
            sender: "tracking-delivery@dhl-parcel-update.top",
            subject: "DHL Express: Failed delivery attempt for parcel #998124",
            body: "Dear Customer,\n\nWe were unable to deliver your package today due to an incorrect shipping address. Click http://dhl.com@track-package.xyz:8080/confirm to re-enter your delivery address and pay the $2.95 redelivery fee.\n\nAction required within 48 hours or package will be returned to sender."
        },
        wire: {
            sender: "ceo.office@executive-urgent-board.click",
            subject: "Confidential: Immediate Wire Transfer Request",
            body: "Are you at your desk?\n\nI am currently in an executive board meeting and need you to execute an urgent wire transfer request of $42,500 to vendor account #88219 before the international banking cutoff today.\n\nDo not discuss this with anyone until the acquisition paperwork is signed. Please confirm as soon as the transfer is processed."
        }
    };

    // Initialize System Status Check
    fetchHealthStatus();

    // Event Listeners
    bodyInput.addEventListener("input", updateCharCount);
    clearBtn.addEventListener("click", resetForm);
    form.addEventListener("submit", handleFormSubmit);

    // Preset Buttons
    document.querySelectorAll(".sample-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            const key = btn.getAttribute("data-sample");
            if (PRESETS[key]) {
                senderInput.value = PRESETS[key].sender;
                subjectInput.value = PRESETS[key].subject;
                bodyInput.value = PRESETS[key].body;
                updateCharCount();
                hideAlert();
                subjectInput.focus();
            }
        });
    });

    /**
     * Checks backend health and model availability.
     */
    async function fetchHealthStatus() {
        try {
            const response = await fetch("/api/health");
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            const data = await response.json();

            // Backend Status
            backendDot.className = "status-dot status-active";
            backendStatusText.textContent = "Backend Online";

            // Model Status
            const mlInfo = data.components?.machine_learning_model;
            if (mlInfo?.is_loaded) {
                modelDot.className = "status-dot status-active";
                modelStatusText.textContent = "ML Model Active";
            } else {
                modelDot.className = "status-dot status-warning";
                modelStatusText.textContent = "Model Not Loaded (DFA Mode)";
            }
        } catch (err) {
            backendDot.className = "status-dot status-danger";
            backendStatusText.textContent = "Backend Offline";
            modelDot.className = "status-dot status-neutral";
            modelStatusText.textContent = "Unknown";
        }
    }

    /**
     * Updates character counter indicator.
     */
    function updateCharCount() {
        const length = bodyInput.value.length;
        charCountEl.textContent = `${length.toLocaleString()} / 50,000 characters`;
    }

    /**
     * Shows error alert banner safely.
     */
    function showAlert(message) {
        formAlertText.textContent = message;
        formAlert.style.display = "flex";
    }

    /**
     * Hides error alert banner.
     */
    function hideAlert() {
        formAlert.style.display = "none";
        formAlertText.textContent = "";
    }

    /**
     * Resets form fields and returns UI to empty state.
     */
    function resetForm() {
        form.reset();
        updateCharCount();
        hideAlert();
        emptyState.style.display = "flex";
        loadingState.style.display = "none";
        populatedResults.style.display = "none";
    }

    /**
     * Handles email analysis form submission.
     */
    async function handleFormSubmit(e) {
        e.preventDefault();
        hideAlert();

        const subject = subjectInput.value.trim();
        const body = bodyInput.value.trim();
        const sender = senderInput.value.trim();

        if (!subject && !body) {
            showAlert("Please enter either an email subject or email body to analyze.");
            subjectInput.focus();
            return;
        }

        // Set UI to loading state
        setLoading(true);

        try {
            const response = await fetch("/api/analyze", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ subject, body, sender })
            });

            const data = await response.json();

            if (!response.ok) {
                showAlert(data.error || `Server returned error status ${response.status}`);
                setLoading(false);
                return;
            }

            // Render Results safely
            renderResults(data);
            fetchHealthStatus(); // Refresh badge in case model loaded

        } catch (err) {
            showAlert(`Network or server connection failed: ${err.message}`);
        } finally {
            setLoading(false);
        }
    }

    /**
     * Toggles button and card loading states.
     */
    function setLoading(isLoading) {
        if (isLoading) {
            analyzeBtn.disabled = true;
            analyzeSpinner.style.display = "inline-block";
            analyzeBtnText.textContent = "Scanning...";
            emptyState.style.display = "none";
            populatedResults.style.display = "none";
            loadingState.style.display = "flex";
        } else {
            analyzeBtn.disabled = false;
            analyzeSpinner.style.display = "none";
            analyzeBtnText.textContent = "Analyze Email";
            loadingState.style.display = "none";
        }
    }

    /**
     * Safely populates the results panel using textContent and DOM methods.
     */
    function renderResults(data) {
        // Show container
        emptyState.style.display = "none";
        loadingState.style.display = "none";
        populatedResults.style.display = "block";

        const riskScore = Number(data.risk_score) || 0;
        const classification = data.final_classification || "INCONCLUSIVE";
        const riskLevel = data.risk_level || "UNKNOWN";

        // Verdict Banner
        riskLevelBadge.textContent = riskLevel;
        finalClassification.textContent = data.classification || classification.replace(/_/g, " ");
        operatingModeTag.textContent = data.operating_mode ? data.operating_mode.replace(/_/g, " ") : "Standard Mode";
        riskScoreValue.textContent = Math.round(riskScore);

        // Banner and Dial Styling
        verdictBanner.className = "verdict-banner";
        scoreCircle.style.borderColor = "";

        if (classification === "PHISHING" || riskScore >= 65) {
            verdictBanner.classList.add("verdict-phishing");
            scoreCircle.style.borderColor = "var(--status-danger)";
            riskLevelBadge.style.backgroundColor = "var(--status-danger)";
            riskLevelBadge.style.color = "#ffffff";
        } else if (classification === "SUSPICIOUS" || (riskScore >= 30 && riskScore < 65)) {
            verdictBanner.classList.add("verdict-suspicious");
            scoreCircle.style.borderColor = "var(--status-warning)";
            riskLevelBadge.style.backgroundColor = "var(--status-warning)";
            riskLevelBadge.style.color = "#000000";
        } else if (classification === "LIKELY_LEGITIMATE") {
            verdictBanner.classList.add("verdict-legitimate");
            scoreCircle.style.borderColor = "var(--status-safe)";
            riskLevelBadge.style.backgroundColor = "var(--status-safe)";
            riskLevelBadge.style.color = "#ffffff";
        } else {
            verdictBanner.classList.add("verdict-inconclusive");
            scoreCircle.style.borderColor = "var(--status-neutral)";
            riskLevelBadge.style.backgroundColor = "var(--status-neutral)";
            riskLevelBadge.style.color = "#ffffff";
        }

        // 1. DFA Card Data
        const dfa = data.dfa || {};
        dfaScoreBadge.textContent = `Score: ${Math.round(dfa.dfa_risk_score || 0)}`;
        dfaTransitionsCount.textContent = (dfa.total_transitions || 0).toLocaleString();
        dfaPhrasesCount.textContent = (dfa.detected_phrases?.length || 0);
        
        let urlAnomaliesCount = 0;
        if (dfa.url_findings) {
            dfa.url_findings.forEach(u => {
                urlAnomaliesCount += (u.anomalies?.length || 0);
            });
        }
        dfaUrlAnomaliesCount.textContent = urlAnomaliesCount;
        dfaSummaryText.textContent = dfa.summary || "DFA scan finished.";

        // 2. ML Card Data
        const ml = data.machine_learning || {};
        if (ml.is_available) {
            mlStatusBadge.textContent = "Active";
            mlStatusBadge.className = "metric-badge badge-ml";
            mlProbabilityValue.textContent = ml.phishing_probability !== null ? `${Math.round(ml.phishing_probability * 100)}%` : "N/A";
            mlConfidenceValue.textContent = ml.confidence !== null ? `${Math.round(ml.confidence * 100)}%` : "N/A";
            mlThresholdValue.textContent = `${Math.round((ml.decision_threshold || 0.5) * 100)}%`;
            mlStatusMessage.textContent = ml.status_message || "Model evaluation successful.";
        } else {
            mlStatusBadge.textContent = "Offline";
            mlStatusBadge.className = "metric-badge";
            mlStatusBadge.style.backgroundColor = "rgba(239, 68, 68, 0.2)";
            mlStatusBadge.style.color = "#fca5a5";
            mlProbabilityValue.textContent = "--";
            mlConfidenceValue.textContent = "--";
            mlThresholdValue.textContent = "--";
            mlStatusMessage.textContent = ml.status_message || "Model not trained yet.";
        }

        // 3. SLM Card Data
        const slm = data.slm || {};
        if (slm.is_available) {
            slmBadge.textContent = "Connected";
            slmStatusText.textContent = slm.message || "Connected to SLM service.";
        } else {
            slmBadge.textContent = "Modular Interface Ready";
            slmStatusText.textContent = slm.message || "SLM interface ready for fine-tuned reasoning model.";
        }

        // 4. Detected Phrases
        phrasesContainer.innerHTML = "";
        const detectedPhrases = dfa.detected_phrases || [];
        if (detectedPhrases.length > 0) {
            phrasesGroup.style.display = "flex";
            detectedPhrases.forEach((p) => {
                const item = document.createElement("div");
                item.className = "phrase-item";

                const name = document.createElement("span");
                name.className = "phrase-name";
                name.textContent = `"${p.pattern}"`;

                const meta = document.createElement("div");
                meta.className = "phrase-meta";

                const category = document.createElement("span");
                category.className = "phrase-category";
                category.style.fontSize = "0.74rem";
                category.style.color = "var(--text-muted)";
                category.textContent = p.category;

                const severity = document.createElement("span");
                severity.className = `severity-pill severity-${p.severity}`;
                severity.textContent = p.severity;

                meta.appendChild(category);
                meta.appendChild(severity);
                item.appendChild(name);
                item.appendChild(meta);
                phrasesContainer.appendChild(item);
            });
        } else {
            phrasesGroup.style.display = "none";
        }

        // 5. Detected URL Findings
        urlsContainer.innerHTML = "";
        const urlFindings = dfa.url_findings || [];
        const flaggedUrls = urlFindings.filter(u => u.anomalies && u.anomalies.length > 0);

        if (flaggedUrls.length > 0) {
            urlsGroup.style.display = "flex";
            flaggedUrls.forEach((u) => {
                const item = document.createElement("div");
                item.className = "url-item";

                // Raw URL rendered as plain text inside code element (NEVER a clickable anchor!)
                const rawUrlSpan = document.createElement("span");
                rawUrlSpan.className = "url-raw-line";
                rawUrlSpan.textContent = `Host: ${u.host || "Unknown"} (URL: ${u.url})`;
                item.appendChild(rawUrlSpan);

                const anomList = document.createElement("ul");
                anomList.className = "url-anom-list";

                u.anomalies.forEach((a) => {
                    const anomItem = document.createElement("li");
                    anomItem.className = "url-anom-item";
                    anomItem.textContent = `• [${a.type} - ${a.severity}]: ${a.explanation}`;
                    anomList.appendChild(anomItem);
                });

                item.appendChild(anomList);
                urlsContainer.appendChild(item);
            });
        } else {
            urlsGroup.style.display = "none";
        }

        // 6. Contributing ML Features
        featuresContainer.innerHTML = "";
        const features = ml.contributing_features || [];
        if (features.length > 0) {
            mlFeaturesGroup.style.display = "flex";
            features.forEach((f) => {
                const tag = document.createElement("span");
                tag.className = `feature-tag ${f.direction === "Phishing" ? "dir-phish" : "dir-legit"}`;
                tag.textContent = `${f.term} (${f.direction})`;
                featuresContainer.appendChild(tag);
            });
        } else {
            mlFeaturesGroup.style.display = "none";
        }

        // 7. Decision Rationale
        reasonsList.innerHTML = "";
        const reasons = data.reasons || [];
        if (reasons.length > 0) {
            reasons.forEach((r) => {
                const li = document.createElement("li");
                li.className = "reason-item";

                const bullet = document.createElement("span");
                bullet.className = "reason-bullet";
                bullet.textContent = "▸";

                const textSpan = document.createElement("span");
                textSpan.textContent = r;

                li.appendChild(bullet);
                li.appendChild(textSpan);
                reasonsList.appendChild(li);
            });
        }

        // 8. Disclaimer
        if (data.disclaimer) {
            disclaimerText.textContent = data.disclaimer;
        }
    }
});
