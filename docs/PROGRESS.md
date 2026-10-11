# Progress Tracker

## Phase 0: Data Verification & Inspection
- [x] Inspect dataset filesystem layout (`run7`, `run8`).
- [x] Verify data schemas and stream properties (`rgb_frames`, `depth_frames`, `imu`, `heading`, `gps_*`, `yolo`).
- [x] Document calibration ray unprojection math ($X, Y, Z$ unit rays) and depth units (uint16 mm).
- [x] Create `docs/PHASE0_REPORT.md` and `docs/DECISIONS.md`.

## Phase 1: Pure-PyTorch Mamba & Pipeline Implementation
- [x] Configuration module with CPU/GPU support (`config.py`).
- [x] Pure-PyTorch Selective State-Space model block (`mamba.py`).
- [x] Multimodal temporal dataset loader with causal timestamp alignment (`dataset.py`).
- [x] Multimodal sequence prediction architecture (`model.py`).
- [x] Full training pipeline with smoke size limits (`train.py`).
- [x] Verified training on real UrbanEgo dataset (loss reduced from 0.011040 to 0.007415, MAE = 0.026486).
- [x] Checkpoint saved (`checkpoint.pt`, ~2.96 MB).

## Phase 2: Transformer Pipeline & Head-to-Head Comparison
- [x] Causal sequence Transformer architecture with zero comments (`transformer_model.py`).
- [x] Hyperparameter scaling in `config.py` (max 100 epochs, early stopping patience = 7, train/val split = 80/20).
- [x] Early stopping pipeline stopping training when validation response degrades (`compare.py`).
- [x] Checkpoints saved for both models (`mamba_checkpoint.pt`, `transformer_checkpoint.pt`).
- [x] Head-to-head comparison chart generated (`comparison_graph.png`).
- [x] Winner: Drishti Mamba achieved superior predictive accuracy (MAE 0.050878 vs Transformer 0.054599 with fewer parameters).
