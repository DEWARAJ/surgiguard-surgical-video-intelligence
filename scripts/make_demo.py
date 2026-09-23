from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from surgiguard.data import SyntheticSurgicalVideo  # noqa: E402
from surgiguard.metrics import risk_events, tip_coordinates  # noqa: E402
from surgiguard.model import TemporalSurgiNet  # noqa: E402


def main() -> None:
    checkpoint = torch.load(ROOT / "artifacts" / "surgiguard_synthetic.pt", map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = TemporalSurgiNet(config["base_channels"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    frames, masks, _, _, _ = SyntheticSurgicalVideo(1, config["image_size"], config["sequence_length"], 2026)[0]
    with torch.no_grad():
        logits, heatmaps = model(frames.unsqueeze(0))
    probabilities = logits.softmax(dim=2)
    prediction = logits.argmax(dim=2)
    uncertainty = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=2) / np.log(3.0)
    tips = tip_coordinates(heatmaps)
    risks = risk_events(prediction, tips, config["risk_distance_pixels"])[0]

    colors = np.asarray([[0, 0, 0], [55, 210, 255], [255, 72, 95]], dtype=np.uint8)
    figure, axes = plt.subplots(3, config["sequence_length"], figsize=(13, 8), constrained_layout=True)
    for t in range(config["sequence_length"]):
        image = frames[t].permute(1, 2, 0).numpy()
        mask_rgb = colors[prediction[0, t].numpy()] / 255.0
        overlay = 0.68 * image + 0.32 * mask_rgb
        tip_x, tip_y = tips[0, t].numpy()

        axes[0, t].imshow(image)
        axes[0, t].set_title(f"Frame {t + 1}")
        axes[1, t].imshow(overlay.clip(0, 1))
        axes[1, t].scatter([tip_x], [tip_y], c="#ffe66d", s=45, marker="x", linewidths=2)
        axes[1, t].set_title("ALERT" if bool(risks[t]) else "MONITOR", color="#ef233c" if bool(risks[t]) else "#198754", fontweight="bold")
        axes[2, t].imshow(uncertainty[0, t].numpy(), cmap="magma", vmin=0, vmax=1)
        axes[2, t].set_title("Predictive entropy")
        for row in range(3):
            axes[row, t].axis("off")

    figure.suptitle("SurgiGuard - Temporal Segmentation, Tip Tracking, and Safety-Zone Monitoring", fontsize=15, fontweight="bold")
    figure.text(0.5, 0.01, "Cyan: instrument  |  Red: protected anatomy  |  Yellow X: predicted instrument tip  |  Synthetic demonstration", ha="center", fontsize=10)
    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    figure.savefig(assets / "demo_dashboard.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    # A lightweight animated artifact makes the temporal behavior easy to inspect in a portfolio.
    animation_frames = []
    for t in range(config["sequence_length"]):
        image = (frames[t].permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
        mask_rgb = colors[prediction[0, t].numpy()]
        blended = (0.68 * image + 0.32 * mask_rgb).clip(0, 255).astype(np.uint8)
        canvas = Image.fromarray(blended).resize((512, 512), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(canvas)
        tip_x, tip_y = tips[0, t].numpy() * 8
        draw.ellipse((tip_x - 9, tip_y - 9, tip_x + 9, tip_y + 9), outline=(255, 230, 109), width=4)
        label = "ALERT" if bool(risks[t]) else "MONITOR"
        draw.rectangle((0, 0, 512, 44), fill=(190, 25, 52) if label == "ALERT" else (25, 130, 86))
        draw.text((18, 11), f"SurgiGuard | Frame {t + 1} | {label}", fill="white")
        animation_frames.append(canvas)
    animation_frames[0].save(
        assets / "surgiguard_demo.gif",
        save_all=True,
        append_images=animation_frames[1:],
        duration=650,
        loop=0,
    )

    history = json.loads((ROOT / "results" / "training_history.json").read_text(encoding="utf-8"))
    fig, ax_left = plt.subplots(figsize=(8, 4.5))
    epochs = [item["epoch"] for item in history]
    ax_left.plot(epochs, [item["train_loss"] for item in history], marker="o", label="Training loss", color="#173b57")
    ax_left.set_xlabel("Epoch")
    ax_left.set_ylabel("Training loss", color="#173b57")
    ax_right = ax_left.twinx()
    ax_right.plot(epochs, [item["mean_foreground_iou"] for item in history], marker="s", label="Validation mIoU", color="#e63946")
    ax_right.set_ylabel("Synthetic validation mIoU", color="#e63946")
    ax_left.grid(alpha=0.25)
    fig.suptitle("SurgiGuard Training History (Synthetic Verification Dataset)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(assets / "training_curves.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
