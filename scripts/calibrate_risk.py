from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from surgiguard.data import SyntheticSurgicalVideo  # noqa: E402
from surgiguard.evaluate import evaluate_model  # noqa: E402
from surgiguard.model import TemporalSurgiNet  # noqa: E402


def main() -> None:
    checkpoint = torch.load(ROOT / "artifacts" / "surgiguard_synthetic.pt", map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = TemporalSurgiNet(config["base_channels"])
    model.load_state_dict(checkpoint["model_state"])
    dataset = SyntheticSurgicalVideo(
        config["validation_sequences"],
        config["image_size"],
        config["sequence_length"],
        config["seed"] + 100_000,
    )
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    sweep = []
    for threshold in range(0, 9):
        scores = evaluate_model(model, loader, torch.device("cpu"), float(threshold))
        sweep.append({
            "threshold_pixels": threshold,
            "risk_precision": scores["risk_precision"],
            "risk_recall": scores["risk_recall"],
            "risk_f1": scores["risk_f1"],
        })
    best = max(sweep, key=lambda row: (row["risk_f1"], row["risk_precision"], -row["threshold_pixels"]))
    report = {
        "evidence_type": "synthetic_only",
        "selection_protocol": "threshold selected on the fixed synthetic validation split",
        "best": best,
        "sweep": sweep,
    }
    output = ROOT / "results" / "risk_threshold_sweep.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
