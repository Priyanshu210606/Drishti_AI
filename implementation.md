# Drishti AI: strict rules

SOURCE OF TRUTH: IMPLEMENTATION.md (incl. §54-71; amendments win on conflict). Never edit it.
At the START of EVERY session and phase, re-read: IMPLEMENTATION.md, docs/PHASE0_REPORT.md, docs/DECISIONS.md, docs/PROGRESS.md (if they exist).

HARDWARE: CPU-ONLY laptop. NO GPU.
- Never run or install anything needing CUDA/GPU. Install CPU PyTorch only (--index-url https://download.pytorch.org/whl/cpu). No mamba_ssm fused kernels; use a verified pure-PyTorch Mamba.
- device=cpu in every config. torch.set_num_threads from config.
- Every script must support config limits (max_runs, max_frames, epochs, batch_size). Default to SMOKE SIZE: <=2 runs, <=200 frames, <=2 epochs, batch <=8.
- Never run full-dataset processing, full training, or benchmarks without my explicit approval.
- Before any command that may take >2 min or >2 GB RAM: state estimate, then ASK.
- If something can't run on CPU: write the code, report "NOT RUN (needs GPU)", move on. Never fake output.

LOOP (every phase, no skipping):
1. READ-BACK (max 8 lines): phase exit criteria; verified facts you rely on + their source file; UNKNOWNs; steps that are CPU-unsafe.
2. PLAN: goal, files to create/modify + why, deps, exact commands, CPU cost estimate. Then WAIT for "approved".
3. IMPLEMENT only this phase.
4. VERIFY with CPU-safe checks only. Paste REAL output.
5. Update docs/PROGRESS.md (+ DECISIONS.md if a decision was made).
6. REPORT: done / tested / actual results / errors / UNKNOWN / blockers.
7. STOP. Wait for "approved: Phase N+1".

HARD BANS: guessing dataset structure, filenames, fields, depth units, or library APIs (inspect, else write UNKNOWN); dummy/random/fake data in the pipeline; invented metrics; future info in inputs; hardcoded paths/thresholds/class ids; silent fallbacks; overwriting existing code without stating the conflict; new deps without checking installed + CPU compatibility; files not needed this phase; utils.py/helpers.py.

If ambiguous or sections conflict: ASK. Never choose silently.
Be terse. No restating the spec. No unrequested summaries.
