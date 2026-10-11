import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from config import DrishtiConfig
from dataset import UrbanEgoDataset
from model import DrishtiMamba

def evaluate_and_plot():
    cfg = DrishtiConfig()
    device = cfg.setup_runtime()

    dataset = UrbanEgoDataset(cfg)
    assert len(dataset) > 0

    model = DrishtiMamba(cfg).to(device)
    ckpt_path = cfg.mamba_checkpoint_path if os.path.exists(cfg.mamba_checkpoint_path) else cfg.checkpoint_path
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_targets = []
    all_preds = []
    all_depths = []
    all_obstacles = []

    with torch.no_grad():
        for i in range(len(dataset)):
            x, y = dataset[i]
            pred = model(x.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
            target = y.numpy()

            all_preds.append(pred)
            all_targets.append(target)
            all_depths.append(x[-1, 14].item() * cfg.depth_clip_meters)
            all_obstacles.append(x[-1, 11].item())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_depths = np.array(all_depths)
    all_obstacles = np.array(all_obstacles)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    steps = np.arange(min(100, len(all_targets)))
    axes[0, 0].plot(steps, all_targets[:100, 0], label="Ground Truth Delta", color="black", linewidth=1.5)
    axes[0, 0].plot(steps, all_preds[:100, 0], label="Predicted Delta", color="red", linestyle="--", linewidth=1.5)
    axes[0, 0].set_title("Ego-Motion Yaw Delta: Ground Truth vs Predicted")
    axes[0, 0].set_xlabel("Time Step (Sequence Index)")
    axes[0, 0].set_ylabel("Delta Angle (rad)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(steps, all_targets[:100, 1], label="GT Pitch", color="blue", linewidth=1.2)
    axes[0, 1].plot(steps, all_preds[:100, 1], label="Pred Pitch", color="cyan", linestyle="--", linewidth=1.2)
    axes[0, 1].plot(steps, all_targets[:100, 2], label="GT Roll", color="green", linewidth=1.2)
    axes[0, 1].plot(steps, all_preds[:100, 2], label="Pred Roll", color="lime", linestyle="--", linewidth=1.2)
    axes[0, 1].set_title("Pitch & Roll Deltas: Ground Truth vs Predicted")
    axes[0, 1].set_xlabel("Time Step")
    axes[0, 1].set_ylabel("Delta Angle (rad)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    errors = np.abs(all_preds - all_targets)
    mean_errors_per_step = np.mean(errors, axis=1)
    axes[1, 0].hist(mean_errors_per_step, bins=25, color="steelblue", edgecolor="black", alpha=0.7)
    axes[1, 0].axvline(np.mean(mean_errors_per_step), color="red", linestyle="dashed", linewidth=1.5, label=f"Mean Error: {np.mean(mean_errors_per_step):.4f}")
    axes[1, 0].set_title("Prediction Absolute Error Distribution")
    axes[1, 0].set_xlabel("Mean Absolute Error (rad)")
    axes[1, 0].set_ylabel("Sample Frequency")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    scatter = axes[1, 1].scatter(all_depths, all_obstacles, c=mean_errors_per_step, cmap="viridis", alpha=0.8, edgecolors="none")
    fig.colorbar(scatter, ax=axes[1, 1], label="Prediction Error (rad)")
    axes[1, 1].set_title("Multimodal Context: Depth Distance vs Obstacle Count")
    axes[1, 1].set_xlabel("Min Depth Distance (meters)")
    axes[1, 1].set_ylabel("Total Detected Obstacles (YOLO)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = "evaluation_graph.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Evaluation graphs successfully saved to {plot_path}")

if __name__ == "__main__":
    evaluate_and_plot()
