document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const toggleBtn = document.getElementById("toggleBtn");
    const toggleBtnText = document.getElementById("toggleBtnText");
    const switchCamBtn = document.getElementById("switchCamBtn");
    const testAudioBtn = document.getElementById("testAudioBtn");
    const openSettingsBtn = document.getElementById("openSettingsBtn");
    const closeSettingsBtn = document.getElementById("closeSettingsBtn");

    const liveAnnouncement = document.getElementById("liveAnnouncement");
    const connectionStatus = document.getElementById("connectionStatus");
    const statusText = document.getElementById("statusText");
    const pathPill = document.getElementById("pathPill");
    const modelPill = document.getElementById("modelPill");
    const speechModePill = document.getElementById("speechModePill");
    const fpsDisplay = document.getElementById("fpsDisplay");

    const videoPreview = document.getElementById("videoPreview");
    const overlayCanvas = document.getElementById("overlayCanvas");
    const cameraStandby = document.getElementById("cameraStandby");

    // Diagnostics Elements
    const diagModel = document.getElementById("diagModel");
    const diagConfidence = document.getElementById("diagConfidence");
    const diagAgreement = document.getElementById("diagAgreement");
    const diagTotalLatency = document.getElementById("diagTotalLatency");
    const diagDetectLatency = document.getElementById("diagDetectLatency");
    const diagMambaLatency = document.getElementById("diagMambaLatency");
    const diagTransLatency = document.getElementById("diagTransLatency");
    const diagObstacleCount = document.getElementById("diagObstacleCount");
    const diagObjectList = document.getElementById("diagObjectList");

    // Settings Modal Elements
    const settingsModal = document.getElementById("settingsModal");
    const verbositySelect = document.getElementById("verbositySelect");
    const speechRate = document.getElementById("speechRate");
    const rateValue = document.getElementById("rateValue");
    const speechVolume = document.getElementById("speechVolume");
    const volValue = document.getElementById("volValue");
    const speechInterval = document.getElementById("speechInterval");
    const intervalValue = document.getElementById("intervalValue");

    // Onboarding Elements
    const onboardingModal = document.getElementById("onboardingModal");
    const acceptOnboardBtn = document.getElementById("acceptOnboardBtn");

    // Initialize Sub-managers
    const camera = new window.CameraManager(videoPreview, overlayCanvas);
    const ws = new window.WebSocketClient("/ws");
    const speech = window.speechEngine;

    let isAssistanceActive = false;

    // Safe LocalStorage helpers
    const safeGetStorage = (key, fallback = null) => {
        try {
            const v = localStorage.getItem(key);
            return v !== null ? v : fallback;
        } catch (e) {
            return fallback;
        }
    };

    const safeSetStorage = (key, val) => {
        try {
            localStorage.setItem(key, val);
        } catch (e) {}
    };

    // Check First-Time Onboarding
    if (!safeGetStorage("drishti_onboarded")) {
        onboardingModal.hidden = false;
        onboardingModal.style.setProperty("display", "flex", "important");
        onboardingModal.classList.remove("is-hidden");
    } else {
        onboardingModal.hidden = true;
        onboardingModal.style.setProperty("display", "none", "important");
        onboardingModal.classList.add("is-hidden");
    }

    const dismissOnboard = () => {
        safeSetStorage("drishti_onboarded", "true");
        onboardingModal.hidden = true;
        onboardingModal.style.setProperty("display", "none", "important");
        onboardingModal.classList.add("is-hidden");
        speech.initAudio();
        speech.playStartCue();
        speech.speak("Welcome to Drishti AI. Ready when you are.", "INFO");
    };

    if (acceptOnboardBtn) {
        acceptOnboardBtn.addEventListener("click", dismissOnboard);
    }
    window.dismissOnboarding = dismissOnboard;

    // Load Persisted Settings
    const savedMode = safeGetStorage("drishti_speech_mode", "normal");
    verbositySelect.value = savedMode;
    speechModePill.textContent = `${savedMode.toUpperCase()} MODE`;

    const savedRate = parseFloat(safeGetStorage("drishti_speech_rate", "1.0"));
    speechRate.value = savedRate;
    rateValue.textContent = `${savedRate.toFixed(1)}x`;
    speech.rate = savedRate;

    const savedVol = parseFloat(safeGetStorage("drishti_speech_vol", "1.0"));
    speechVolume.value = savedVol;
    volValue.textContent = `${Math.round(savedVol * 100)}%`;
    speech.volume = savedVol;

    // WebSocket Status Callback
    ws.onStatusChangeCallback = (status, text) => {
        statusText.textContent = text;
        connectionStatus.className = `status-pill status-${status}`;
    };

    // WebSocket Incoming Inference Message Handler
    ws.onMessageCallback = (res) => {
        if (!isAssistanceActive) return;

        // 1. Update Spoken Announcement Text
        if (res.navigation && res.navigation.text) {
            liveAnnouncement.textContent = res.navigation.text;

            // Trigger Text-to-Speech if anti-spam manager approved
            if (res.navigation.should_speak) {
                speech.speak(res.navigation.text, res.navigation.priority);
            }
        }

        // 2. Update Path Clearance Pill
        const pStatus = res.path_status || "clear";
        pathPill.className = `path-pill path-${pStatus === "clear" ? "clear" : (pStatus === "blocked" ? "blocked" : "caution")}`;
        pathPill.textContent = `PATH: ${pStatus.toUpperCase()}`;

        // 3. Update Model Selection Pill
        if (res.gating) {
            const mName = (res.gating.selected_model || "DUAL").toUpperCase();
            const confPct = Math.round((res.gating.confidence || 0) * 100);
            modelPill.textContent = `MODEL: ${mName} (${confPct}%)`;
        }

        // 4. Draw Obstacle Bounding Boxes on Camera Canvas
        if (res.objects) {
            camera.drawDetections(res.objects);
        }

        // 5. Update Diagnostics
        fpsDisplay.textContent = `${res.fps.toFixed(1)} FPS`;
        diagModel.textContent = res.gating ? res.gating.selected_model.toUpperCase() : "-";
        diagConfidence.textContent = res.gating ? `${Math.round(res.gating.confidence * 100)}%` : "-";
        diagAgreement.textContent = res.gating ? `${(res.gating.agreement_score * 100).toFixed(0)}%` : "-";

        if (res.latencies) {
            diagTotalLatency.textContent = `${res.latencies.total_ms} ms`;
            diagDetectLatency.textContent = `${res.latencies.detection_ms} ms`;
            diagMambaLatency.textContent = `${res.latencies.mamba_ms} ms`;
            diagTransLatency.textContent = `${res.latencies.transformer_ms} ms`;
        }

        diagObstacleCount.textContent = (res.objects ? res.objects.length : 0);
        if (res.objects && res.objects.length > 0) {
            diagObjectList.innerHTML = res.objects.map(o =>
                `<li><strong>${o.label}</strong>: ${o.distance_m}m (${o.direction.replace("_", " ")})</li>`
            ).join("");
        } else {
            diagObjectList.innerHTML = "<li>None detected</li>";
        }
    };

    // Connect WebSocket on Load
    ws.connect();

    // Toggle Assistance (Start / Stop)
    toggleBtn.addEventListener("click", async () => {
        speech.initAudio();

        if (!isAssistanceActive) {
            // Start Assistance
            try {
                toggleBtnText.textContent = "INITIALIZING CAMERA...";
                toggleBtn.disabled = true;

                await camera.startCamera();
                cameraStandby.style.display = "none";

                isAssistanceActive = true;
                toggleBtn.className = "btn btn-danger btn-large";
                toggleBtnText.textContent = "STOP ASSISTANCE";
                toggleBtn.setAttribute("aria-label", "Stop Assistance");
                toggleBtn.disabled = false;

                connectionStatus.className = "status-pill status-active";
                statusText.textContent = "Assistance Active";

                speech.playStartCue();
                speech.speak("Assistance active. Analyzing path ahead.", "HIGH");

                // Start Streaming Frames to WebSocket at 12 FPS
                camera.startFrameCapture(12, (frameBlob) => {
                    ws.sendFrame(frameBlob);
                });

            } catch (err) {
                console.error("Camera startup error:", err);
                toggleBtn.disabled = false;
                toggleBtn.className = "btn btn-primary btn-large";
                toggleBtnText.textContent = "START ASSISTANCE";

                let errorMsg = "Camera access was denied or unavailable.";
                let speakMsg = "Camera error. Please grant camera permission.";

                if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
                    errorMsg = "Camera permission was denied. Please click the camera lock icon in your address bar and allow access.";
                    speakMsg = "Camera permission denied. Please allow camera access.";
                } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
                    errorMsg = "No webcam detected. Please connect a camera and refresh the page.";
                    speakMsg = "No camera detected. Please connect a webcam.";
                } else if (err.name === "NotReadableError" || err.name === "TrackStartError") {
                    errorMsg = "Camera is currently being used by another application (Zoom/Teams/browser tab). Please close other apps and try again.";
                    speakMsg = "Camera is in use by another application.";
                } else if (err.message) {
                    errorMsg = `Camera error: ${err.message}`;
                }

                liveAnnouncement.textContent = errorMsg;
                speech.speak(speakMsg, "CRITICAL");
            }
        } else {
            // Stop Assistance
            isAssistanceActive = false;
            camera.stopCamera();
            cameraStandby.style.display = "block";

            toggleBtn.className = "btn btn-primary btn-large";
            toggleBtnText.textContent = "START ASSISTANCE";
            toggleBtn.setAttribute("aria-label", "Start Assistance");

            connectionStatus.className = "status-pill status-ready";
            statusText.textContent = "System Ready";
            liveAnnouncement.textContent = "Assistance paused. Press Start to resume.";
            pathPill.className = "path-pill path-clear";
            pathPill.textContent = "PATH: READY";

            speech.playStopCue();
            speech.speak("Assistance stopped.", "HIGH");
        }
    });

    // Switch Front / Rear Camera
    switchCamBtn.addEventListener("click", async () => {
        if (isAssistanceActive) {
            const newFacing = await camera.switchCamera();
            speech.speak(`Switched to ${newFacing === "environment" ? "rear" : "front"} camera.`, "INFO");
        } else {
            camera.currentFacingMode = (camera.currentFacingMode === "environment") ? "user" : "environment";
            speech.speak(`Camera set to ${camera.currentFacingMode === "environment" ? "rear" : "front"}.`, "INFO");
        }
    });

    // Test Audio
    testAudioBtn.addEventListener("click", () => {
        speech.playStartCue();
        speech.speak("Audio check. Drishti navigation speech output is operating normally.", "HIGH");
    });

    // Settings Modal
    openSettingsBtn.addEventListener("click", () => {
        settingsModal.hidden = false;
        settingsModal.style.display = "flex";
        settingsModal.classList.remove("is-hidden");
    });

    closeSettingsBtn.addEventListener("click", () => {
        settingsModal.hidden = true;
        settingsModal.style.display = "none";
        settingsModal.classList.add("is-hidden");
        const newMode = verbositySelect.value;
        localStorage.setItem("drishti_speech_mode", newMode);
        speechModePill.textContent = `${newMode.toUpperCase()} MODE`;
        ws.sendConfig({ speech_mode: newMode });
        speech.speak("Settings saved.", "INFO");
    });

    speechRate.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        rateValue.textContent = `${val.toFixed(1)}x`;
        speech.rate = val;
        localStorage.setItem("drishti_speech_rate", val.toString());
    });

    speechVolume.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        volValue.textContent = `${Math.round(val * 100)}%`;
        speech.volume = val;
        localStorage.setItem("drishti_speech_vol", val.toString());
    });

    speechInterval.addEventListener("input", (e) => {
        intervalValue.textContent = `${parseFloat(e.target.value).toFixed(1)}s`;
    });
});
