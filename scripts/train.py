from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from surgiguard.data import SyntheticSurgicalVideo  # noqa: E402
from surgiguard.evaluate import evaluate_model  # noqa: E402
from surgiguard.losses import combined_loss  # noqa: E402
from surgiguard.model import TemporalSurgiNet, parameter_count  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synthetic_mvp.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = SyntheticSurgicalVideo(
        config["train_sequences"], config["image_size"], config["sequence_length"], seed
    )
    validation_data = SyntheticSurgicalVideo(
        config["validation_sequences"], config["image_size"], config["sequence_length"], seed + 100_000
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_data, batch_size=config["batch_size"], shuffle=False)

    model = TemporalSurgiNet(config["base_channels"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    history = []
    for epoch in range(config["epochs"]):
        model.train()
        epoch_losses = []
        for frames, masks, tip_heatmaps, _, _ in train_loader:
            frames, masks, tip_heatmaps = frames.to(device), masks.to(device), tip_heatmaps.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, predicted_tips = model(frames)
            loss, components = combined_loss(
                logits, predicted_tips, masks, tip_heatmaps, config["loss_weights"]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))
        metrics = evaluate_model(model, validation_loader, device, config["risk_distance_pixels"])
        record = {
            "epoch": epoch + 1,
            "train_loss": sum(epoch_losses) / len(epoch_losses),
            **metrics,
        }
        history.append(record)
        print(json.dumps(record))

    artifacts = ROOT / "artifacts"
    results = ROOT / "results"
    artifacts.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "config": config, "parameter_count": parameter_count(model)},
        artifacts / "surgiguard_synthetic.pt",
    )
    final = {
        "evidence_type": "synthetic_only",
        "device": str(device),
        "parameter_count": parameter_count(model),
        "validation_sequences": len(validation_data),
        **history[-1],
    }
    (results / "metrics.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    (results / "training_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

