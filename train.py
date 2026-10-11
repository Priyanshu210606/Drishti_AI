import os
import sys
import time
import argparse
import copy
import torch

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from config import DrishtiConfig
from dataset import UrbanEgoDataset
from model import DrishtiMamba

def train():
    parser = argparse.ArgumentParser(description="Drishti AI - Fast Mamba Training")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs to train (default: 1)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (default: 128)")
    parser.add_argument("--full", action="store_true", help="Use full 128k dataset instead of 5,000 samples/epoch")
    parser.add_argument("--fresh", action="store_true", help="Start training fresh from epoch 0")
    args = parser.parse_args()

    cfg = DrishtiConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.fresh:
        cfg.resume = False
    if args.full:
        cfg.samples_per_epoch = None

    device = cfg.setup_runtime()
    print(f"=== Drishti AI Mamba Training Pipeline ===", flush=True)
    print(f"Device: {device} | Epochs this run: {cfg.epochs} | Batch size: {cfg.batch_size}", flush=True)

    dataset = UrbanEgoDataset(cfg)
    total_samples = len(dataset)

    if cfg.samples_per_epoch is not None and cfg.samples_per_epoch < total_samples:
        indices = torch.randperm(total_samples, generator=torch.Generator().manual_seed(42))[:cfg.samples_per_epoch]
        train_ds = Subset(dataset, indices)
        print(f"Fast Mode: {len(train_ds)} samples drawn randomly from {total_samples} total.", flush=True)
    else:
        train_ds = dataset
        print(f"Full Mode: {len(train_ds)} samples.", flush=True)

    loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    model = DrishtiMamba(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()

    start_epoch = 0
    best_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())

    # Resume if checkpoint exists
    target_ckpt = cfg.mamba_checkpoint_path if os.path.exists(cfg.mamba_checkpoint_path) else cfg.checkpoint_path
    if os.path.exists(target_ckpt) and getattr(cfg, "resume", True):
        try:
            ckpt = torch.load(target_ckpt, map_location=device, weights_only=False)
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
                best_weights = copy.deepcopy(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except Exception:
                    pass
            start_epoch = int(ckpt.get("epoch", 0))
            best_loss = float(ckpt.get("best_val_loss", ckpt.get("loss", float("inf"))))
            print(f"Resumed from {target_ckpt} (Previous Epoch: {start_epoch}, Best Loss: {best_loss:.6f})", flush=True)
        except Exception as e:
            print(f"Starting fresh (checkpoint notice: {e})", flush=True)

    epochs_to_run = cfg.epochs
    for step in range(1, epochs_to_run + 1):
        current_epoch = start_epoch + step
        model.train()
        total_loss = 0.0
        batches = 0
        epoch_start = time.time()

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / max(1, batches)
        elapsed = time.time() - epoch_start

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_weights = copy.deepcopy(model.state_dict())
            flag = "* NEW BEST *"
        else:
            flag = ""

        print(f"Epoch {current_epoch} (step {step}/{epochs_to_run}) - Loss: {avg_loss:.6f} {flag} [{elapsed:.1f}s]", flush=True)

        # Save checkpoint immediately
        torch.save({
            "epoch": current_epoch,
            "model_state_dict": best_weights if best_weights is not None else model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "best_val_loss": best_loss,
            "config": cfg.__dict__
        }, cfg.mamba_checkpoint_path)
        print(f"  -> Checkpoint updated on disk: {cfg.mamba_checkpoint_path}", flush=True)

    model.eval()
    with torch.no_grad():
        sample_x, sample_y = dataset[0]
        test_pred = model(sample_x.unsqueeze(0).to(device)).squeeze(0).cpu()
        print("\nSelf-check Inference:", flush=True)
        print(f"  Ground Truth Delta: {sample_y.numpy()}", flush=True)
        print(f"  Predicted Delta:    {test_pred.numpy()}", flush=True)
        error = torch.mean(torch.abs(test_pred - sample_y)).item()
        print(f"  Mean Absolute Error: {error:.6f}", flush=True)

if __name__ == "__main__":
    train()
