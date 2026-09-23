from __future__ import annotations

import torch
import torch.nn.functional as functional


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, classes: int = 3) -> torch.Tensor:
    probabilities = logits.softmax(dim=2)
    one_hot = functional.one_hot(targets, num_classes=classes).permute(0, 1, 4, 2, 3).float()
    intersection = (probabilities[:, :, 1:] * one_hot[:, :, 1:]).sum(dim=(0, 1, 3, 4))
    denominator = (probabilities[:, :, 1:] + one_hot[:, :, 1:]).sum(dim=(0, 1, 3, 4))
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return 1.0 - dice.mean()


def temporal_consistency_loss(logits: torch.Tensor) -> torch.Tensor:
    probabilities = logits.softmax(dim=2)
    if probabilities.shape[1] < 2:
        return probabilities.new_tensor(0.0)
    return (probabilities[:, 1:] - probabilities[:, :-1]).abs().mean()


def combined_loss(
    segmentation_logits: torch.Tensor,
    tip_heatmaps: torch.Tensor,
    masks: torch.Tensor,
    target_tip_heatmaps: torch.Tensor,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    batch, time, classes, height, width = segmentation_logits.shape
    ce = functional.cross_entropy(
        segmentation_logits.reshape(batch * time, classes, height, width),
        masks.reshape(batch * time, height, width),
    )
    dice = soft_dice_loss(segmentation_logits, masks, classes)
    tip = functional.binary_cross_entropy(tip_heatmaps, target_tip_heatmaps)
    temporal = temporal_consistency_loss(segmentation_logits)
    total = (
        weights["cross_entropy"] * ce
        + weights["dice"] * dice
        + weights["tip"] * tip
        + weights["temporal"] * temporal
    )
    return total, {
        "cross_entropy": float(ce.detach()),
        "dice_loss": float(dice.detach()),
        "tip_loss": float(tip.detach()),
        "temporal_loss": float(temporal.detach()),
    }

