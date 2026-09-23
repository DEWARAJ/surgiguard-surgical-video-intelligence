from __future__ import annotations

from collections import defaultdict

import torch

from .metrics import (
    binary_scores,
    class_metrics,
    expected_calibration_error,
    mean_tip_error,
    risk_events,
    temporal_jitter,
    tip_coordinates,
)


@torch.no_grad()
def evaluate_model(model, loader, device: torch.device, risk_threshold: float) -> dict[str, float]:
    model.eval()
    values: dict[str, list[float]] = defaultdict(list)
    all_risk_predictions, all_risk_targets = [], []
    for frames, masks, target_heatmaps, tip_xy, risk_targets in loader:
        frames, masks = frames.to(device), masks.to(device)
        target_heatmaps, tip_xy = target_heatmaps.to(device), tip_xy.to(device)
        logits, heatmaps = model(frames)
        prediction = logits.argmax(dim=2)
        for key, value in class_metrics(prediction, masks).items():
            values[key].append(value)
        values["tip_error_px"].append(mean_tip_error(heatmaps, tip_xy))
        values["expected_calibration_error"].append(expected_calibration_error(logits, masks))
        values["temporal_jitter"].append(temporal_jitter(prediction))
        predicted_risk = risk_events(prediction, tip_coordinates(heatmaps), risk_threshold)
        all_risk_predictions.append(predicted_risk.cpu())
        all_risk_targets.append(risk_targets.bool())
    result = {key: sum(items) / len(items) for key, items in values.items()}
    result.update(binary_scores(torch.cat(all_risk_predictions), torch.cat(all_risk_targets)))
    return result

