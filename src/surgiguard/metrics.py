from __future__ import annotations

import math

import torch


def class_metrics(prediction: torch.Tensor, target: torch.Tensor, classes: int = 3) -> dict[str, float]:
    result: dict[str, float] = {}
    for class_id, name in ((1, "instrument"), (2, "protected_anatomy")):
        pred = prediction == class_id
        truth = target == class_id
        intersection = (pred & truth).sum().item()
        union = (pred | truth).sum().item()
        total = pred.sum().item() + truth.sum().item()
        result[f"{name}_iou"] = intersection / union if union else 1.0
        result[f"{name}_dice"] = 2.0 * intersection / total if total else 1.0
    result["mean_foreground_iou"] = (result["instrument_iou"] + result["protected_anatomy_iou"]) / 2.0
    return result


def tip_coordinates(heatmaps: torch.Tensor) -> torch.Tensor:
    batch, time, _, height, width = heatmaps.shape
    flat_indices = heatmaps.reshape(batch, time, -1).argmax(dim=-1)
    y = flat_indices // width
    x = flat_indices % width
    return torch.stack([x, y], dim=-1).float()


def mean_tip_error(predicted_heatmaps: torch.Tensor, target_xy: torch.Tensor) -> float:
    predicted = tip_coordinates(predicted_heatmaps)
    return float(torch.linalg.vector_norm(predicted - target_xy, dim=-1).mean())


def expected_calibration_error(logits: torch.Tensor, target: torch.Tensor, bins: int = 10) -> float:
    probabilities = logits.softmax(dim=2)
    confidence, prediction = probabilities.max(dim=2)
    correct = prediction.eq(target)
    ece = logits.new_tensor(0.0)
    boundaries = torch.linspace(0.0, 1.0, bins + 1, device=logits.device)
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            accuracy = correct[selected].float().mean()
            mean_confidence = confidence[selected].mean()
            ece += selected.float().mean() * (mean_confidence - accuracy).abs()
    return float(ece)


def temporal_jitter(prediction: torch.Tensor) -> float:
    if prediction.shape[1] < 2:
        return 0.0
    return float(prediction[:, 1:].ne(prediction[:, :-1]).float().mean())


def risk_events(prediction: torch.Tensor, tip_xy: torch.Tensor, threshold: float) -> torch.Tensor:
    """Predict proximity risk from protected-anatomy pixels and instrument tip."""
    batch, time, height, width = prediction.shape
    events = torch.zeros((batch, time), dtype=torch.bool, device=prediction.device)
    for batch_id in range(batch):
        for time_id in range(time):
            protected = torch.nonzero(prediction[batch_id, time_id] == 2, as_tuple=False)
            if protected.numel() == 0:
                continue
            tip_x, tip_y = tip_xy[batch_id, time_id]
            distance = torch.sqrt((protected[:, 1] - tip_x) ** 2 + (protected[:, 0] - tip_y) ** 2).min()
            events[batch_id, time_id] = distance <= threshold
    return events


def binary_scores(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction, target = prediction.bool(), target.bool()
    tp = (prediction & target).sum().item()
    fp = (prediction & ~target).sum().item()
    fn = (~prediction & target).sum().item()
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"risk_precision": precision, "risk_recall": recall, "risk_f1": f1}

