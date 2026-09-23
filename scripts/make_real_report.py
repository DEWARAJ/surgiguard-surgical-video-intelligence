from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from surgiguard.cholecseg8k import CLASS_NAMES, CholecSeg8kSequences  # noqa: E402
from surgiguard.model import SurgicalSegNet  # noqa: E402

PALETTE = np.asarray([
    [20, 20, 24], [220, 110, 80], [165, 65, 70], [245, 170, 85], [238, 204, 90],
    [80, 205, 240], [210, 150, 220], [175, 20, 45], [90, 220, 150], [40, 150, 240],
    [110, 205, 95], [150, 105, 210], [245, 120, 195],
], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    return parser.parse_args()


def load_model(name: str, temporal: bool):
    checkpoint = torch.load(ROOT / "artifacts" / f"cholecseg8k_{name}.pt", map_location="cpu", weights_only=True)
    model = SurgicalSegNet(checkpoint["config"]["base_channels"], len(CLASS_NAMES), temporal=temporal)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["config"]


def main() -> None:
    args = parse_args()
    report = json.loads((ROOT / "results" / "cholecseg8k_ablation.json").read_text(encoding="utf-8"))
    frame_model, config = load_model("frame_only", False)
    temporal_model, _ = load_model("temporal", True)
    dataset = CholecSeg8kSequences(
        args.data_root, "test", image_size=tuple(config["image_size"]),
        sequence_length=config["sequence_length"], stride=config["stride"],
        max_sequences=config["test_sequences"], seed=config["seed"],
    )
    worst = report["experiments"][0]["test"]["lowest_foreground_iou_sequences"][:4]
    by_key = {dataset[index][2]: index for index in range(len(dataset))}
    samples = [dataset[by_key[item["key"]]] for item in worst]

    figure, axes = plt.subplots(4, 4, figsize=(14, 10), constrained_layout=True)
    for column, (frames, masks, key) in enumerate(samples):
        with torch.no_grad():
            frame_prediction = frame_model(frames.unsqueeze(0)).argmax(dim=2)[0, -1].numpy()
            temporal_prediction = temporal_model(frames.unsqueeze(0)).argmax(dim=2)[0, -1].numpy()
        image = frames[-1].permute(1, 2, 0).numpy()
        truth = masks[-1].numpy()
        truth = np.where(truth == 255, 0, truth)
        axes[0, column].imshow(image)
        axes[1, column].imshow(PALETTE[truth])
        axes[2, column].imshow(PALETTE[frame_prediction])
        axes[3, column].imshow(PALETTE[temporal_prediction])
        axes[0, column].set_title(key.split("/")[-1], fontsize=9)
        for row in range(4):
            axes[row, column].axis("off")
    for row, label in enumerate(("Real frame", "Ground truth", "Frame-only", "ConvGRU temporal")):
        axes[row, 0].set_ylabel(label, fontsize=11, fontweight="bold")
    figure.suptitle("SurgiGuard Cross-Video Failure Analysis — CholecSeg8k Test Videos", fontweight="bold")
    figure.text(0.5, 0.002, "Video12/video27 held out | 13 classes | CC BY-NC-SA 4.0 | Failure cases selected by binary foreground IoU", ha="center", fontsize=9)
    figure.savefig(ROOT / "assets" / "cholecseg8k_failure_analysis.png", dpi=170, bbox_inches="tight")
    plt.close(figure)

    names = [item["experiment"] for item in report["experiments"]]
    validation = [item["best_validation_mean_foreground_iou"] for item in report["experiments"]]
    test = [item["test"]["mean_foreground_iou"] for item in report["experiments"]]
    flicker = [item["test"]["static_region_flicker"] for item in report["experiments"]]
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(x - 0.18, validation, 0.36, label="Validation", color="#1d6f93")
    axes[0].bar(x + 0.18, test, 0.36, label="Held-out test", color="#e05a47")
    axes[0].set_xticks(x, ["Frame-only", "ConvGRU"])
    axes[0].set_ylabel("Mean foreground IoU")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)
    axes[1].bar(x, flicker, 0.55, color=["#1d6f93", "#e05a47"])
    axes[1].set_xticks(x, ["Frame-only", "ConvGRU"])
    axes[1].set_ylabel("Static-region flicker (lower is better)")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Real-Data Ablation: Temporal Context Did Not Generalize", fontweight="bold")
    fig.tight_layout()
    fig.savefig(ROOT / "assets" / "cholecseg8k_ablation.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
