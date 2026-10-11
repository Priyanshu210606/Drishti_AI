# Drishti AI: Real-Time Assistive Vision Navigation

**Drishti AI** is an assistive-vision and spatial-guidance system engineered to help blind and visually impaired users safely navigate their environment. Through a continuous camera stream (front or rear camera), the system performs real-time object detection, metric distance estimation, spatial angle positioning, and dual-model sequence motion forecasting (**Pure-PyTorch Mamba** + **Causal Transformer**) with **confidence-aware model gating**. Actionable navigation instructions are generated and communicated to the user via prioritized, anti-spammed speech.

> 📖 **Developer & AI Agent Handover Guide:** See [PROJECT_HANDOVER.md](PROJECT_HANDOVER.md) for a full architectural breakdown, training history, benchmark results, runbook, and AI agent guidelines.

---

## Architecture Overview

```
                      +-------------------+
                      |   CAMERA INPUT    |
                      | (Front / Rear 30p)|
                      +---------+---------+
                                |
                   Latest-Frame Bounded Buffer
                                |
             +------------------+------------------+
             |                                     |
             v                                     v
   +--------------------+               +--------------------+
   |  OBJECT DETECTOR   |               |  TEMPORAL BUFFER   |
   | (SSDLite MobileNet)|               |  (16-frame window) |
   +---------+----------+               +----------+---------+
             |                                     |
             v                                     v
   +--------------------+               +----------+----------+
   |  SPATIAL DEPTH &   |               |          |          |
   |  DIRECTION ENGINE  |               v          v          |
   | (Meters & Degrees) |             MAMBA   TRANSFORMER     |
   +---------+----------+               |          |          |
             |                          +-----+----+          |
             |                                |               |
             +-----------------+--------------+               |
                               |                              |
                               v                              |
                    +--------------------+                    |
                    |  CONFIDENCE-AWARE  |                    |
                    |   GATING MODULE    |                    |
                    | (Consensus/Safety) |                    |
                    +----------+---------+                    |
                               |                              |
                               v                              |
                    +--------------------+                    |
                    |  SPATIAL PATH &    |                    |
                    |  CORRIDOR REASONING|                    |
                    +----------+---------+                    |
                               |                              |
                               v                              |
                    +--------------------+                    |
                    | NATURAL NAVIGATION |                    |
                    | MESSAGE GENERATOR  |                    |
                    +----------+---------+                    |
                               |                              |
                               v                              |
                    +--------------------+                    |
                    |   SPEECH MANAGER   |                    |
                    | (Anti-Spam & Cooldown)                  |
                    +----------+---------+                    |
                               |                              |
                               v                              |
                    +--------------------+                    |
                    |  TEXT-TO-SPEECH    |                    |
                    |   (Web Speech API) |                    |
                    +--------------------+                    |
                               |                              |
                               v                              |
                       USER HEARS GUIDANCE:                   |
              "Pole ahead, about 1.8 meters away,             |
                    slightly to your right."                  |
```

---

## Core Capabilities

1. **Continuous Asynchronous Inference**: Decoupled producer-consumer pipeline with a single-slot latest-frame queue. Camera capture never blocks, and stale frames are dropped cleanly.
2. **Dual-Model Inference (Mamba + Transformer)**:
   - **Drishti Mamba**: Pure-PyTorch Selective State-Space model ($S6$ scan) capturing temporal continuous trajectories with lower validation MAE ($0.0508$).
   - **Drishti Transformer**: Causal sequence attention architecture ($0.0546$ MAE) with fast parallel attention.
3. **Confidence-Aware Model Gating**:
   - Computes inter-model agreement score.
   - Boosts confidence under high consensus ($>85\%$).
   - Triggers conservative safety blending when models diverge ($>0.40$ rad difference).
   - Seamless degraded-mode fallback if either model encounters an exception.
4. **Metric Distance & Spatial Positioning**:
   - Computes metric distance in meters using camera optical geometry and object height priors.
   - Categorizes horizontal angle into natural directional sectors: `far_left`, `left`, `slightly_left`, `center`, `slightly_right`, `right`, `far_right`.
5. **Speech Anti-Spam & Priority Queue**:
   - Priority levels: `CRITICAL` (immediate interrupt), `HIGH`, `MEDIUM`, `LOW`, `INFO`.
   - Prevents repetitive vocal announcements unless an obstacle changes, gets significantly closer ($>0.45$m), or path clearance changes.
   - Periodic reassurance reminders for a clear path every 8 seconds.
6. **Accessibility-First Web Application**:
   - WCAG AAA high-contrast dark theme with large touch targets ($\ge 64$px).
   - Screen-reader friendly with semantic HTML and ARIA live regions (`aria-live="assertive"`).
   - Front and rear camera switching (`facingMode: "environment"` and `"user"`).
   - Audio earcon cues for start, stop, camera switch, and hazard detection.
   - Three verbosity modes: **Brief**, **Normal**, and **Detailed**.

---

## Project Structure

```
DrishtiAI/
├── app/
│   ├── config.py                 # Central settings & hyperparameters
│   ├── main.py                   # FastAPI lifespan application entrypoint
│   ├── api/
│   │   ├── health.py             # GET /health system status
│   │   ├── websocket.py          # Duplex /ws real-time streaming endpoint
│   │   └── routes.py             # Route aggregator
│   ├── inference/
│   │   ├── detector.py           # SSDLite MobileNetV3 object detector
│   │   ├── depth.py              # Metric distance & horizontal angle estimator
│   │   ├── temporal_buffer.py    # Rolling 16-frame sequence buffer with angle wrapping
│   │   ├── mamba_inference.py    # Drishti Mamba runner & confidence calibration
│   │   ├── transformer_inference.py # Drishti Transformer runner
│   │   ├── confidence.py         # Reliability & consensus calculator
│   │   ├── gating.py             # Confidence-aware model gating & safety policy
│   │   └── pipeline.py           # End-to-end async inference pipeline
│   ├── navigation/
│   │   ├── reasoning.py          # Corridor safety & path clearance reasoning
│   │   ├── message_generator.py  # Natural language instruction synthesis
│   │   └── speech_manager.py     # Stateful anti-spam rate limiter
│   └── models/
│       ├── loader.py             # Startup model registry & warmup
│       └── schemas.py            # Pydantic data contracts
├── frontend/
│   ├── index.html                # Accessible high-contrast UI
│   ├── styles.css                # Accessible styling & large buttons
│   ├── app.js                    # Core frontend application controller
│   ├── camera.js                 # MediaDevices camera manager & canvas overlays
│   ├── speech.js                 # Web Speech API & audio tone synthesizer
│   └── websocket.js              # Resilient WebSocket client with auto-reconnect
├── tests/
│   └── test_pipeline.py          # Unit & safety scenario test suite
├── mamba_checkpoint.pt           # Trained Drishti Mamba weights
├── transformer_checkpoint.pt     # Trained Drishti Transformer weights
├── requirements.txt              # Production Python dependencies
├── Dockerfile                    # Container definition
├── docker-compose.yml            # Container orchestration
└── README.md
```

---

## Getting Started

### 1. Prerequisites
- Python 3.10+
- Webcam or mobile browser with camera support

### 2. Installation
```bash
pip install -r requirements.txt
```

### 3. Run the Server
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in any web browser.

### 4. Running with Docker
```bash
docker-compose up --build
```

### 5. Deploying to Vercel

This repository includes Vercel configuration for the FastAPI application, model checkpoints, and Python 3.12 runtime.

1. Fork this repository to your GitHub account and push these deployment files to the fork.
2. Import the fork in [Vercel](https://vercel.com/new). Vercel should detect the FastAPI app and install dependencies from `requirements.txt`.
3. Do not set `DEVICE` in Vercel; the app automatically selects CUDA when PyTorch detects an available GPU, otherwise it uses CPU. Vercel Functions currently run this app on CPU.
4. Add `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` as an environment variable so the PyTorch runtime can use Vercel's large-function support if it exceeds the standard Python bundle limit.
5. Deploy, then open the HTTPS URL Vercel provides. Allow camera access in the browser.

Vercel's Hobby plan is intended for personal, non-commercial projects. WebSocket support is currently in beta, and Hobby function invocations are limited to five minutes. The browser reconnects when a connection closes, but a reconnect starts a new inference session and clears its temporal history. Vercel usage quotas and function memory limits also apply.

The deployed site is public and sends camera frames to the hosted server for inference. Do not use it with sensitive video or rely on its output as a substitute for a cane, guide dog, or personal safety awareness.

---

## Running Verification Tests
Execute the comprehensive automated test suite verifying angle wrapping, temporal buffering, spatial depth, confidence gating, and speech anti-spam:

```bash
python -m unittest discover tests
```

---

## Safety & Accessibility Disclaimer
> [!IMPORTANT]
> **Drishti AI provides AI-generated environmental navigation guidance and may make mistakes.**  
> It is an assistive aid designed to complement mobility awareness and must **NOT** replace a white cane, guide dog, or personal safety awareness in hazardous, traffic-heavy, or unfamiliar environments.
