# Architecture & Engineering Decisions

## 1. Pure-PyTorch Mamba (No CUDA Fused Kernels)
- **Context:** Hardware constraint is CPU-only (no GPU). Standard `mamba_ssm` requires CUDA C++ kernels and won't compile or run on CPU.
- **Decision:** Implement pure-PyTorch selective state-space model (S6 scan) using native PyTorch vector operations. Discretization and recurrent state update run natively on CPU without external C extensions.

## 2. Temporal Alignment via Monotonic Timestamp Matching
- **Context:** Sensor streams operate at heterogeneous frequencies (RGB at ~21 fps, depth at ~4.8 fps, IMU at ~14 Hz, GPS at ~1 Hz).
- **Decision:** Use causal backward nearest-match indexing on `ts_unix_ns`. For each frame $t$, sensor values are sampled from the most recent timestamp $\le t$ to strictly prevent future information leakage.

## 3. Depth Sampling & Unit Unprojection
- **Context:** Full unprojection of 320x288 depth images into point clouds per frame consumes memory and CPU cycles during training.
- **Decision:** Compute depth summary statistics (minimum obstacle distance, 10th percentile, 50th percentile) using unit ray calibration matrices $X, Y, Z = \text{ray} \times (\text{depth\_mm} / 1000.0)$. This provides dense spatial distance awareness to the model with minimal CPU overhead.

## 4. Hardware Amendment (GPU Acceleration)
- **Context:** User updated hardware specification: NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM, Driver 617.14, CUDA 13.4).
- **Decision:** Switch model and training pipeline to GPU-first (`cuda` with automatic fallback to `cpu`). Enable float32 / float16 tensor computation on CUDA. Batch size scaled for 4GB VRAM.

## 5. Smoke Size Enforcement
- **Context:** Training budgets must stay within safe bounds.
- **Decision:** Config limits: `max_runs <= 2`, `max_frames <= 200`, `epochs <= 5`, `batch_size <= 8`, `device = 'cuda' if torch.cuda.is_available() else 'cpu'`.
