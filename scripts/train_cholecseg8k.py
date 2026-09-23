from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from surgiguard.cholecseg8k import CLASS_NAMES, CholecSeg8kSequences, split_manifest  # noqa: E402
from surgiguard.model import SurgicalSegNet, parameter_count  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cholecseg8k_real.json")
    parser.add_argument("--data-root", required=True)
    return parser.parse_args()


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probabilities = logits.softmax(dim=2)
    valid = target.ne(255)
    safe_target = target.masked_fill(~valid, 0)
    one_hot = torch.nn.functional.one_hot(safe_target, len(CLASS_NAMES)).permute(0, 1, 4, 2, 3).float()
    valid = valid.unsqueeze(2)
    probabilities = probabilities * valid
    one_hot = one_hot * valid
    dims = (0, 1, 3, 4)
    intersection = (probabilities * one_hot).sum(dims)
    denominator = probabilities.sum(dims) + one_hot.sum(dims)
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


@torch.no_grad()
def evaluate(model, loader, device: torch.device) -> dict[str, object]:
    model.eval()
    intersections = torch.zeros(len(CLASS_NAMES), dtype=torch.float64)
    unions = torch.zeros(len(CLASS_NAMES), dtype=torch.float64)
    changed, total_temporal = 0, 0
    flicker, stable_pixels = 0, 0
    failures = []
    for frames, masks, keys in loader:
        prediction = model(frames.to(device, non_blocking=True)).argmax(dim=2).cpu()
        valid = masks.ne(255)
        for class_id in range(len(CLASS_NAMES)):
            pred_class = prediction.eq(class_id) & valid
            true_class = masks.eq(class_id) & valid
            intersections[class_id] += (pred_class & true_class).sum()
            unions[class_id] += (pred_class | true_class).sum()
        if prediction.shape[1] > 1:
            changed += prediction[:, 1:].ne(prediction[:, :-1]).sum().item()
            total_temporal += prediction[:, 1:].numel()
            stable = valid[:, 1:] & valid[:, :-1] & masks[:, 1:].eq(masks[:, :-1])
            flicker += (prediction[:, 1:].ne(prediction[:, :-1]) & stable).sum().item()
            stable_pixels += stable.sum().item()
        sample_intersection = ((prediction > 0) & (masks > 0) & valid).flatten(1).sum(dim=1).float()
        sample_union = (((prediction > 0) | (masks > 0)) & valid).flatten(1).sum(dim=1).clamp_min(1).float()
        failures.extend((float(i / u), key) for i, u, key in zip(sample_intersection, sample_union, keys))
    per_class = {
        name: float(intersections[i] / unions[i]) if unions[i] else None
        for i, name in enumerate(CLASS_NAMES)
    }
    foreground = [value for name, value in per_class.items() if name != "background" and value is not None]
    return {
        "mean_foreground_iou": sum(foreground) / len(foreground),
        "per_class_iou": per_class,
        "raw_prediction_change": changed / total_temporal if total_temporal else 0.0,
        "static_region_flicker": flicker / stable_pixels if stable_pixels else 0.0,
        "lowest_foreground_iou_sequences": [
            {"key": key, "foreground_iou": score} for score, key in sorted(failures)[:12]
        ],
    }


def train_experiment(name, temporal, config, data_root, device):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    settings = dict(image_size=tuple(config["image_size"]), sequence_length=config["sequence_length"],
                    stride=config["stride"], seed=config["seed"])
    train_data = CholecSeg8kSequences(data_root, "train", max_sequences=config["train_sequences"],
                                      augment=True, **settings)
    validation_data = CholecSeg8kSequences(data_root, "validation",
                                           max_sequences=config["validation_sequences"], **settings)
    test_data = CholecSeg8kSequences(data_root, "test", max_sequences=config["test_sequences"], **settings)
    generator = torch.Generator().manual_seed(config["seed"])
    loader_settings = {
        "batch_size": config["batch_size"],
        "num_workers": config.get("num_workers", 0),
        "pin_memory": device.type == "cuda",
        "persistent_workers": config.get("num_workers", 0) > 0,
    }
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_settings)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_settings)
    test_loader = DataLoader(test_data, shuffle=False, **loader_settings)

    model = SurgicalSegNet(config["base_channels"], len(CLASS_NAMES), temporal=temporal).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=1e-4)
    history, best_score, best_state = [], -1.0, None
    start = time.perf_counter()
    for epoch in range(config["epochs"]):
        model.train()
        losses = []
        for frames, masks, _ in train_loader:
            frames = frames.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(frames)
            loss = torch.nn.functional.cross_entropy(
                logits.flatten(0, 1), masks.flatten(0, 1), ignore_index=255
            )
            loss = loss + config["dice_weight"] * soft_dice_loss(logits, masks)
            if temporal and logits.shape[1] > 1:
                probabilities = logits.softmax(dim=2)
                temporal_loss = (probabilities[:, 1:] - probabilities[:, :-1]).abs().mean()
                loss = loss + config.get("temporal_weight", 0.0) * temporal_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = evaluate(model, validation_loader, device)
        record = {"epoch": epoch + 1, "train_loss": sum(losses) / len(losses), **validation}
        history.append(record)
        print(json.dumps({"experiment": name, "epoch": epoch + 1,
                          "train_loss": record["train_loss"],
                          "validation_miou": record["mean_foreground_iou"]}), flush=True)
        if validation["mean_foreground_iou"] > best_score:
            best_score = validation["mean_foreground_iou"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    test = evaluate(model, test_loader, device)
    artifact = ROOT / "artifacts" / f"cholecseg8k_{name}.pt"
    torch.save({"model_state": best_state, "config": config, "classes": CLASS_NAMES}, artifact)
    return {"experiment": name, "temporal": temporal, "parameter_count": parameter_count(model),
            "training_seconds": time.perf_counter() - start,
            "best_validation_mean_foreground_iou": best_score, "test": test, "history": history}


def main() -> None:
    args = parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = split_manifest(args.data_root, image_size=tuple(config["image_size"]),
                              sequence_length=config["sequence_length"], stride=config["stride"],
                              seed=config["seed"])
    experiments = [train_experiment("frame_only", False, config, args.data_root, device),
                   train_experiment("temporal", True, config, args.data_root, device)]
    report = {"evidence_type": "real_cholecseg8k_subset", "license": "CC BY-NC-SA 4.0",
              "device": str(device), "split_manifest": manifest,
              "sample_limits": {"train": config["train_sequences"],
                                "validation": config["validation_sequences"],
                                "test": config["test_sequences"]},
              "experiments": experiments}
    (ROOT / "results" / "cholecseg8k_ablation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
