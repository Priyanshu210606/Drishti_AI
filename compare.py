import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import time
import copy
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np

from config import DrishtiConfig
from dataset import UrbanEgoDataset
from model import DrishtiMamba
from transformer_model import DrishtiTransformer

def get_epoch_loaders(dataset, cfg, seed=None):
    total_samples = len(dataset)
    if cfg.samples_per_epoch is not None and cfg.samples_per_epoch < total_samples:
        train_count = min(cfg.samples_per_epoch, int(total_samples * (1 - cfg.val_split)))
        val_count = min(getattr(cfg, "val_samples", 1000), int(total_samples * cfg.val_split))
        gen = torch.Generator().manual_seed(seed) if seed is not None else None
        indices = torch.randperm(total_samples, generator=gen)
        train_indices = indices[:train_count]
        val_indices = indices[train_count : train_count + val_count]
        train_ds = Subset(dataset, train_indices)
        val_ds = Subset(dataset, val_indices)
    else:
        val_size = max(1, int(total_samples * cfg.val_split))
        train_size = total_samples - val_size
        gen = torch.Generator().manual_seed(42)
        train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, val_loader

def train_with_early_stopping(model, dataset, cfg, device, checkpoint_path, target_epochs=100):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    start_epoch = 0
    epochs_no_improve = 0
    train_losses = []
    val_losses = []

    # Resume from existing checkpoint if available
    if os.path.exists(checkpoint_path) and getattr(cfg, "resume", True):
        try:
            if os.path.getsize(checkpoint_path) > 0:
                ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
                if "model_state_dict" in ckpt:
                    model.load_state_dict(ckpt["model_state_dict"])
                    best_weights = copy.deepcopy(ckpt["model_state_dict"])
                if "optimizer_state_dict" in ckpt:
                    try:
                        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                    except Exception:
                        pass
                start_epoch = int(ckpt.get("epoch", 0))
                train_losses = list(ckpt.get("train_losses", []))
                val_losses = list(ckpt.get("val_losses", []))
                best_val_loss = min(val_losses) if val_losses else float(ckpt.get("best_val_loss", float("inf")))
                print(f"[{model.__class__.__name__}] Resumed from checkpoint (Previous Epoch: {start_epoch}/{target_epochs}, Best Val Loss: {best_val_loss:.6f})", flush=True)
        except Exception as e:
            print(f"[{model.__class__.__name__}] Note: starting fresh ({e})", flush=True)

    if start_epoch >= target_epochs:
        print(f"[{model.__class__.__name__}] Checkpoint has already reached {start_epoch} epochs (target: {target_epochs}). Use --epochs {start_epoch + 10} to train further or --fresh to start from epoch 1.", flush=True)
        return model, train_losses, val_losses, best_val_loss

    print(f"[{model.__class__.__name__}] Training from epoch {start_epoch + 1} to {target_epochs} (1 epoch at a time)...", flush=True)

    for current_epoch in range(start_epoch + 1, target_epochs + 1):
        t0 = time.time()
        train_loader, val_loader = get_epoch_loaders(dataset, cfg, seed=current_epoch * 100)

        # 1. Train 1 epoch
        model.train()
        train_loss = 0.0
        train_batches = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1
        avg_train = train_loss / max(1, train_batches)
        train_losses.append(avg_train)

        # 2. Validate 1 epoch
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred = model(bx)
                loss = criterion(pred, by)
                val_loss += loss.item()
                val_batches += 1
        avg_val = val_loss / max(1, val_batches)
        val_losses.append(avg_val)

        # 3. Track best weights
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            flag = "* NEW BEST *"
        else:
            epochs_no_improve += 1
            flag = f"(no gain for {epochs_no_improve} ep)"

        elapsed = time.time() - t0
        print(f"[{model.__class__.__name__}] Epoch {current_epoch:3d}/{target_epochs} - Train: {avg_train:.6f} | Val: {avg_val:.6f} | Best Val: {best_val_loss:.6f} {flag} [{elapsed:.1f}s]", flush=True)

        # 4. Save checkpoint immediately after EVERY single epoch
        torch.save({
            "epoch": current_epoch,
            "model_state_dict": best_weights if best_weights is not None else model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "config": cfg.__dict__
        }, checkpoint_path)

        # 5. Early stopping check
        if epochs_no_improve >= cfg.patience:
            print(f"[{model.__class__.__name__}] Early stopping triggered at epoch {current_epoch} (no improvement for {cfg.patience} epochs). Restoring best model weights.", flush=True)
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, train_losses, val_losses, best_val_loss

def evaluate_model(model, val_loader, device):
    model.eval()
    criterion = nn.MSELoss()
    all_preds = []
    all_targets = []

    t0 = time.time()
    with torch.no_grad():
        for bx, by in val_loader:
            bx = bx.to(device)
            preds = model(bx).cpu()
            all_preds.append(preds.numpy())
            all_targets.append(by.numpy())
    total_samples = max(1, sum(len(p) for p in all_preds))
    latency_ms = ((time.time() - t0) / total_samples) * 1000.0

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    mse = float(np.mean((all_preds - all_targets) ** 2))
    mae = float(np.mean(np.abs(all_preds - all_targets)))
    params = sum(p.numel() for p in model.parameters())

    return {
        "mse": mse,
        "mae": mae,
        "latency_ms": latency_ms,
        "params": params,
        "preds": all_preds,
        "targets": all_targets
    }

def main():
    parser = argparse.ArgumentParser(description="Drishti AI - Fast Resumable Dual Model Training (100 Epochs)")
    parser.add_argument("--epochs", type=int, default=None, help="Target total epochs (default from config: 100)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: 128)")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience (default: 20)")
    parser.add_argument("--full", action="store_true", help="Use full 128k dataset (slower) instead of 5,000 samples/epoch")
    parser.add_argument("--fresh", action="store_true", help="Start training from epoch 0 instead of resuming")
    args = parser.parse_args()

    cfg = DrishtiConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.patience is not None:
        cfg.patience = args.patience
    if args.fresh:
        cfg.resume = False
    if args.full:
        cfg.samples_per_epoch = None
        cfg.val_samples = None

    device = cfg.setup_runtime()

    print("==========================================================", flush=True)
    print("    DRISHTI AI - FAST RESUMABLE TRAINING (100 EPOCHS)    ", flush=True)
    print(f"Device: {device} | Target Epochs: {cfg.epochs} | Batch size: {cfg.batch_size}", flush=True)
    print("==========================================================", flush=True)

    dataset = UrbanEgoDataset(cfg)
    total_samples = len(dataset)
    print(f"Dataset active: {total_samples} total sequences cached from all 6 runs.", flush=True)

    # 1. Train Drishti Mamba
    print(f"\n--- [1/2] Training Drishti Mamba (Target: {cfg.epochs} Epochs) ---", flush=True)
    mamba = DrishtiMamba(cfg).to(device)
    mamba, mamba_tr_losses, mamba_val_losses, mamba_best_val = train_with_early_stopping(
        mamba, dataset, cfg, device, cfg.mamba_checkpoint_path, target_epochs=cfg.epochs
    )

    # 2. Train Drishti Transformer
    print(f"\n--- [2/2] Training Drishti Transformer (Target: {cfg.epochs} Epochs) ---", flush=True)
    transformer = DrishtiTransformer(cfg).to(device)
    transformer, trans_tr_losses, trans_val_losses, trans_best_val = train_with_early_stopping(
        transformer, dataset, cfg, device, cfg.transformer_checkpoint_path, target_epochs=cfg.epochs
    )

    # 3. Final Evaluation
    _, eval_val_loader = get_epoch_loaders(dataset, cfg, seed=9999)
    mamba_eval = evaluate_model(mamba, eval_val_loader, device)
    transformer_eval = evaluate_model(transformer, eval_val_loader, device)

    print("\n=================== FINAL COMPARISON RESULTS ===================", flush=True)
    print(f"{'Metric':<25} | {'Drishti Mamba':<18} | {'Drishti Transformer':<18}", flush=True)
    print("-" * 68, flush=True)
    print(f"{'Best Validation MSE':<25} | {mamba_eval['mse']:<18.6f} | {transformer_eval['mse']:<18.6f}", flush=True)
    print(f"{'Mean Absolute Error (MAE)':<25} | {mamba_eval['mae']:<18.6f} | {transformer_eval['mae']:<18.6f}", flush=True)
    print(f"{'Inference Latency':<25} | {mamba_eval['latency_ms']:<15.2f} ms | {transformer_eval['latency_ms']:<15.2f} ms", flush=True)
    m_p = f"{mamba_eval['params']:,}"
    t_p = f"{transformer_eval['params']:,}"
    print(f"{'Parameters':<25} | {m_p:<18} | {t_p:<18}", flush=True)
    print(f"{'Total Epochs Trained':<25} | {len(mamba_tr_losses):<18} | {len(trans_tr_losses):<18}", flush=True)

    winner = "Drishti Mamba" if mamba_eval['mae'] < transformer_eval['mae'] else "Drishti Transformer"
    print(f"\nWINNER: {winner} achieved superior predictive accuracy.", flush=True)

    # 4. Generate / Update comparison graph
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # Loss curves
    m_epochs = range(1, len(mamba_val_losses) + 1)
    t_epochs = range(1, len(trans_val_losses) + 1)
    axes[0, 0].plot(m_epochs, mamba_val_losses, label=f"Mamba Val (Best: {mamba_best_val:.4f})", color="blue", linewidth=1.8, marker="o", markersize=3)
    axes[0, 0].plot(t_epochs, trans_val_losses, label=f"Transformer Val (Best: {trans_best_val:.4f})", color="crimson", linewidth=1.8, marker="s", markersize=3)
    axes[0, 0].set_title("Validation Loss Curves (Cumulative 100 Epochs)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("MSE Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Predictions vs Ground Truth
    steps = np.arange(min(60, len(mamba_eval['targets'])))
    axes[0, 1].plot(steps, mamba_eval['targets'][:steps.size, 0], label="Ground Truth Delta", color="black", linewidth=2.0)
    axes[0, 1].plot(steps, mamba_eval['preds'][:steps.size, 0], label="Mamba Pred", color="blue", linestyle="--", linewidth=1.5)
    axes[0, 1].plot(steps, transformer_eval['preds'][:steps.size, 0], label="Transformer Pred", color="crimson", linestyle=":", linewidth=1.5)
    axes[0, 1].set_title("Validation Predictions vs Ground Truth (Yaw Delta)")
    axes[0, 1].set_xlabel("Validation Sequence Index")
    axes[0, 1].set_ylabel("Delta Angle (rad)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Error Residual Distributions
    mamba_err = np.abs(mamba_eval['preds'] - mamba_eval['targets']).mean(axis=1)
    trans_err = np.abs(transformer_eval['preds'] - transformer_eval['targets']).mean(axis=1)
    axes[1, 0].hist(mamba_err, bins=20, alpha=0.6, label=f"Mamba (Mean: {mamba_eval['mae']:.4f})", color="blue", edgecolor="black")
    axes[1, 0].hist(trans_err, bins=20, alpha=0.6, label=f"Transformer (Mean: {transformer_eval['mae']:.4f})", color="crimson", edgecolor="black")
    axes[1, 0].set_title("Error Residual Distributions")
    axes[1, 0].set_xlabel("Absolute Error (rad)")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # KPI comparison
    categories = ["Val MSE (x100)", "Val MAE (x100)", "Latency (ms)"]
    mamba_vals = [mamba_eval['mse'] * 100, mamba_eval['mae'] * 100, mamba_eval['latency_ms']]
    trans_vals = [transformer_eval['mse'] * 100, transformer_eval['mae'] * 100, transformer_eval['latency_ms']]
    x_pos = np.arange(len(categories))
    width = 0.35
    axes[1, 1].bar(x_pos - width/2, mamba_vals, width, label="Mamba", color="blue", alpha=0.8)
    axes[1, 1].bar(x_pos + width/2, trans_vals, width, label="Transformer", color="crimson", alpha=0.8)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(categories)
    axes[1, 1].set_title("Key Performance Indicators")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = "comparison_graph.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"\nComparison chart saved to {chart_path}", flush=True)

if __name__ == "__main__":
    main()
