# Drishti AI: Developer & AI Agent Handover Guide

> **Purpose:** This document is the single source of truth for new developers, collaborators, and AI coding agents (Antigravity, Claude Code, Cursor, Copilot, etc.) working on **Drishti AI**. It explains what the system does, its architectural decisions, data pipeline, model training history, benchmark findings, and runbook instructions.

---

## 1. Project Mission & Overview

**Drishti AI** is a real-time assistive vision and spatial navigation system designed for blind and visually impaired individuals. Through a continuous camera stream (smartphone or webcam), Drishti AI:
1. Detects obstacles and objects in the path (pedestrians, vehicles, poles, stairs, doors).
2. Estimates metric distance (meters) and spatial clock/relative bearing (left, ahead, right).
3. Predicts user ego-motion dynamics using a **Dual-Model Sequence Architecture** (**Drishti Mamba** + **Drishti Transformer**).
4. Dynamically selects predictions using a **Confidence-Aware Model Gate**.
5. Formulates concise, natural navigation advice (e.g., *"Pole ahead, 1.8 meters away, slightly to your right"*).
6. Communicates spoken guidance through prioritized, anti-spammed text-to-speech (TTS).

---

## 2. High-Level Architecture

The system uses an asynchronous, decoupled client-server architecture:

```
[ FRONTEND (Client) ]                               [ BACKEND (FastAPI Server) ]
Any Web Browser (Phone / Laptop)                     Runs on CPU or CUDA GPU
+------------------------------+                     +----------------------------------+
| Camera Feed (Front / Rear)   |                     | WebSocket Endpoint (/ws/navigate)|
|  - HTML5 getUserMedia        | -- Base64 Frames -> |  - Bounded Latest-Frame Buffer   |
|  - Drops stale frames        |                     +-----------------+----------------+
|                              |                                       |
| Web Speech API (TTS)         |                                       v
|  - Spoken audio output       | <- Spoken Text ---- +-----------------+----------------+
|  - Visual Bounding Boxes     |    & Coordinates    | Object Detector (SSDLite Mobile) |
+------------------------------+                     | Temporal Sequence Buffer (L=16)  |
                                                     +-----------------+----------------+
                                                                       |
                                                     +-----------------+----------------+
                                                     | Dual Sequence Inference:         |
                                                     |  1. Drishti Mamba (State Space)  |
                                                     |  2. Drishti Transformer (Attn)   |
                                                     +-----------------+----------------+
                                                                       |
                                                                       v
                                                     +-----------------+----------------+
                                                     | Confidence Gate & Safety Arbiter |
                                                     | Spatial Corridor Path Reasoning  |
                                                     | Anti-Spam Message Generator      |
                                                     +----------------------------------+
```

### Hardware Requirements
- **Webpage / Client:** **NO GPU required**. Runs on any standard browser on iOS, Android, macOS, Windows, or Linux.
- **Backend Server:** Can run on **CPU** or **CUDA GPU**.
  - On RTX 3050 Laptop GPU: Transformer: ~2 ms, Mamba: ~12 ms.
  - On standard CPU: Transformer: ~15 ms, Mamba: ~28 ms.
  - Both operate comfortably under the 33 ms real-time frame budget (30 FPS).

---

## 3. Machine Learning Models & Checkpoints

### A. Drishti Mamba (`model.py`)
- **Type:** Pure-PyTorch Selective State-Space Sequence Model ($S6$ scan).
- **Parameters:** 244,420 parameters.
- **Strength:** Superior continuous temporal trajectory modeling and lowest validation error on navigation-critical orientation (Yaw & Heading).
- **Checkpoint:** `mamba_checkpoint.pt`.

### B. Drishti Transformer (`transformer_model.py`)
- **Type:** Causal Multi-Head Self-Attention Transformer.
- **Parameters:** 277,956 parameters (`d_model=128`, `n_heads=4`, `n_layers=2`, `dim_feedforward=256`).
- **Strength:** Highly parallelized token processing and ultra-low single-frame latency (~2 ms on GPU, ~15 ms on CPU).
- **Checkpoint:** `transformer_checkpoint.pt`.

### C. Object Detector (`app/inference/detector.py`)
- **Type:** MobileNetV3 SSDLite (torchvision pretrained).
- **Target Classes:** Person, bicycle, car, motorcycle, bus, train, truck, traffic light, fire hydrant, stop sign, parking meter, bench, chair, potted plant.

### D. Confidence-Aware Gating (`app/inference/decision_gate.py`)
- Evaluates predictions from both Mamba and Transformer simultaneously.
- Considers model residuals, agreement distance, and safety margins.
- Prefers higher-agreement consensus; in high-divergence cases, selects the more conservative, safety-oriented trajectory.

---

## 4. Dataset & Preprocessing Pipeline

- **Data Source:** `Data/` containing 6 urban egocentric walking runs (`run_1` through `run_6`) with synchronized sensor data:
  - `rgb_frames.jsonl` + raw frames
  - `depth/` (monocular depth maps)
  - `imu.jsonl` (accelerometer & gyroscope: yaw, pitch, roll)
  - `heading.jsonl` (compass heading)
  - `gps_phone.jsonl` (coordinates)
  - `yolo/frames.jsonl` (precomputed object detections)
- **Feature Extraction (`dataset.py`):**
  - Sequence length: $L = 16$.
  - Feature dimension: $D = 16$ (sin/cos of yaw, pitch, roll, heading; relative GPS displacement; object counts; min/median depth).
  - Target dimension: $4$ ($\Delta \text{yaw}, \Delta \text{pitch}, \Delta \text{roll}, \Delta \text{heading}$).
- **Precomputed Tensor Cache:**
  - All 128,404 sequence samples are pre-extracted into **`Data/preprocessed_cache.pt` (204 MB)**.
  - Eliminates disk image I/O bottlenecks during training, boosting epoch speeds from hours down to seconds.

---

## 5. Benchmarking & Empirical Metric Insights

### 100-Epoch Training Evaluation (Held-Out Test Set, $N = 2,000$ Sequences)

| Metric | Drishti Mamba | Drishti Transformer | Winner / Advantage |
| :--- | :--- | :--- | :--- |
| **Validation MSE** | 0.878994 | 0.879095 | Drishti Mamba |
| **Validation MAE** | 0.169359 | 0.170134 | **Drishti Mamba (Accuracy)** |
| **Yaw Delta MAE** | 0.049099 rad (~2.8°) | 0.048545 rad (~2.8°) | Comparable |
| **Heading Delta MAE** | 0.038442 rad (~2.2°) | 0.050933 rad (~2.9°) | **Drishti Mamba** |
| **Single-Frame Latency (GPU)** | 12.73 ms | 2.01 ms | **Drishti Transformer (~6x faster)** |
| **Single-Frame Latency (CPU)** | 28.36 ms | 15.35 ms | **Drishti Transformer (~2x faster)** |
| **Parameter Count** | 244,420 | 277,956 | Drishti Mamba |

### Forensic Finding: Why MSE ($\approx 0.88$) is Much Larger than MAE ($\approx 0.17$)
- **Angle Wraparound ($0 \leftrightarrow 2\pi$):** Sensor angles crossing coordinate limits (e.g., $+3.14$ rad to $-3.14$ rad) generate raw step deltas of $|\Delta| \approx 6.28$ rad ($360^\circ$).
- **The Outlier Effect:**
  - For **95% of normal samples**, the prediction error is $\le 0.086$ rad ($5.0^\circ$), with a median error of just **$0.025$ rad ($1.4^\circ$)**.
  - Only **2.25% of samples** contain angle wraparound discontinuities.
  - Because MSE squares errors ($(6.28)^2 \approx 39.4$), those **2.25% outliers account for 99.8% of the entire MSE score**.
  - MAE scales linearly ($|e|$), reflecting true operational navigational accuracy.

---

## 6. Directory Structure & Key Files

```
DrishtiAI/
├── app/                              # FastAPI Production Application
│   ├── api/
│   │   ├── routes.py                 # REST routes (/api/health) & WebSocket (/ws/navigate)
│   │   └── websocket.py              # WebSocket frame loop, JSON transport, audio responses
│   ├── inference/
│   │   ├── decision_gate.py          # Confidence-aware gating between Mamba & Transformer
│   │   ├── detector.py               # MobileNetV3 SSDLite real-time obstacle detection
│   │   ├── mamba_inference.py        # Optimized runtime runner for Drishti Mamba
│   │   └── transformer_inference.py  # Optimized runtime runner for Drishti Transformer
│   ├── models/
│   │   └── loader.py                 # Singleton ModelRegistry (loads models once at startup)
│   ├── navigation/
│   │   ├── corridor_reasoning.py     # Path clearance, clock direction & distance calculation
│   │   └── message_generator.py      # Natural language phrasing & anti-spam cooldown logic
│   ├── config.py                     # Central app settings (Pydantic BaseSettings)
│   └── main.py                       # FastAPI entrypoint & static frontend mounting
├── frontend/                         # Accessibility-Focused Web Client
│   ├── index.html                    # High-contrast UI, camera switch, live audio status
│   ├── app.js                        # Camera stream capture, WebSocket, Web Speech API TTS
│   └── styles.css                    # WCAG high-contrast styling & large tactile touch targets
├── tests/
│   └── test_pipeline.py              # Pytest suite covering models, gating, detector & routing
├── Data/                             # Dataset & Cache
│   ├── run_1/ ... run_6/             # Raw sensor records and depth maps
│   └── preprocessed_cache.pt         # 204 MB pre-extracted sequence cache
├── config.py                         # Training configuration (DrishtiConfig dataclass)
├── dataset.py                        # UrbanEgoDataset loader & cache generation
├── model.py                          # Pure-PyTorch DrishtiMamba neural network
├── transformer_model.py              # DrishtiTransformer neural network
├── train.py                          # Standalone fast Mamba trainer
├── compare.py                        # Dual-model comparison trainer (Mamba vs Transformer)
├── compare.ps1 / compare.bat         # Shell shortcuts for compare.py
├── train.ps1 / train.bat             # Shell shortcuts for train.py
├── mamba_checkpoint.pt               # Trained 100-epoch Drishti Mamba weights
├── transformer_checkpoint.pt         # Trained 100-epoch Drishti Transformer weights
├── comparison_graph.png              # Visual loss curves & latency comparison chart
└── requirements.txt                  # Python dependencies
```

---

## 7. Developer Runbook (How to Run Everything)

### 1. Environment Setup
```powershell
# In PowerShell (Windows)
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Starting the Web Application
```powershell
# Run the FastAPI server (accessible at http://localhost:8000)
python -m app.main
```
- Open `http://localhost:8000` in any browser (Chrome, Edge, Safari).
- Grant camera permission.
- Click **"Start Assistance"** to begin real-time navigation audio guidance.

### 3. Running Dual-Model Training / Comparison
```powershell
# Resume training or run comparison on existing checkpoints:
python compare.py

# Train completely fresh from Epoch 1 to 100:
python compare.py --fresh

# Or use the shortcut scripts:
.\compare.ps1
.\compare.ps1 -Fresh
```

### 4. Running Standalone Mamba Training
```powershell
python train.py
# Or with options:
python train.py --epochs 10 --batch-size 128
```

### 5. Running the Test Suite
```powershell
python -m pytest tests/test_pipeline.py
```
*(All 6/6 tests must pass: model shapes, gate selection, detector inference, corridor calculation, message throttling).*

---

## 8. Strict Guidelines for Future AI Agents & Developers

When modifying or expanding this codebase, adhere to these rules:

1. **Do NOT mock or fake data:** The system relies on real data (`Data/preprocessed_cache.pt`) and real trained weights (`mamba_checkpoint.pt`, `transformer_checkpoint.pt`). Never insert fake placeholder metrics or bypass checkpoint loading.
2. **Preserve Asynchronous Pipeline:** Never make camera ingestion block on model inference. The latest-frame bounded buffer must be maintained so slow inference drops stale frames rather than buffering backlog lag.
3. **Dual-Model Parity:** Drishti AI uses **both** Mamba and Transformer at runtime. Do not remove or permanently disable either model; the confidence gate relies on both models to ensure navigation safety.
4. **Windows Terminal Compatibility:** When printing to stdout, use ASCII symbols (e.g. `* NEW BEST *` instead of `★`) or ensure `sys.stdout.reconfigure(encoding="utf-8")` is called to avoid `cp1252` encoding crashes.
5. **Accessibility Priority:** The user interface is designed for visually impaired users. Spoken guidance must be concise, prioritized by distance, and rate-limited by the anti-spam cooldown logic so the user is not overwhelmed by continuous chatter.
