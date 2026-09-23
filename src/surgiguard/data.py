from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SequenceSample:
    frames: torch.Tensor
    masks: torch.Tensor
    tip_heatmaps: torch.Tensor
    tip_xy: torch.Tensor
    risk: torch.Tensor


def _tip_heatmap(size: int, x: float, y: float, sigma: float = 2.2) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    heat = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * sigma**2))
    return heat.astype(np.float32)


def _point_to_ellipse_distance(x: float, y: float, box: tuple[int, int, int, int]) -> float:
    left, top, right, bottom = box
    cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
    rx, ry = max((right - left) / 2.0, 1.0), max((bottom - top) / 2.0, 1.0)
    normalized = math.sqrt(((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2)
    return abs(normalized - 1.0) * min(rx, ry) if normalized > 1.0 else 0.0


class SyntheticSurgicalVideo(Dataset):
    """Deterministic procedural video sequences for software verification.

    Classes: 0 background/tissue, 1 instrument, 2 protected anatomy.
    """

    def __init__(self, sequences: int, image_size: int = 64, sequence_length: int = 4, seed: int = 0):
        self.sequences = sequences
        self.image_size = image_size
        self.sequence_length = sequence_length
        self.seed = seed

    def __len__(self) -> int:
        return self.sequences

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        rng = random.Random(self.seed + index * 7919)
        size = self.image_size
        organ_w = rng.randint(size // 4, size // 3)
        organ_h = rng.randint(size // 5, size // 3)
        organ_cx = rng.randint(size // 2, int(size * 0.78))
        organ_cy = rng.randint(int(size * 0.32), int(size * 0.72))
        organ_box = (
            organ_cx - organ_w // 2,
            organ_cy - organ_h // 2,
            organ_cx + organ_w // 2,
            organ_cy + organ_h // 2,
        )

        start_y = rng.randint(size // 4, 3 * size // 4)
        end_x = rng.randint(int(size * 0.45), int(size * 0.88))
        end_y = rng.randint(size // 4, 3 * size // 4)
        moving_toward = rng.random() > 0.35

        frames, masks, heatmaps, tips, risks = [], [], [], [], []
        for t in range(self.sequence_length):
            alpha = t / max(self.sequence_length - 1, 1)
            if not moving_toward:
                alpha = 1.0 - alpha
            tip_x = int(size * 0.18 + alpha * (end_x - size * 0.18))
            tip_y = int(start_y + alpha * (end_y - start_y))
            base_x = -4
            base_y = start_y + rng.randint(-1, 1)

            background = np.zeros((size, size, 3), dtype=np.uint8)
            yy, xx = np.mgrid[0:size, 0:size]
            texture = 12 * np.sin(xx / 7.0) + 9 * np.cos(yy / 9.0)
            noise = np.random.default_rng(self.seed + index * 97 + t).normal(0, 4, (size, size))
            background[..., 0] = np.clip(116 + texture + noise, 0, 255)
            background[..., 1] = np.clip(46 + 0.35 * texture + noise, 0, 255)
            background[..., 2] = np.clip(55 + 0.25 * texture + noise, 0, 255)
            image = Image.fromarray(background).filter(ImageFilter.GaussianBlur(radius=0.7))
            draw = ImageDraw.Draw(image)
            draw.ellipse(organ_box, fill=(178, 91, 99), outline=(228, 154, 157), width=2)
            draw.line((base_x, base_y, tip_x, tip_y), fill=(185, 194, 207), width=max(3, size // 18))
            draw.ellipse((tip_x - 3, tip_y - 3, tip_x + 3, tip_y + 3), fill=(235, 244, 248))
            if rng.random() > 0.55:
                draw.ellipse((size - 16, 5, size - 5, 16), fill=(245, 238, 225))

            mask_image = Image.new("L", (size, size), color=0)
            mask_draw = ImageDraw.Draw(mask_image)
            mask_draw.ellipse(organ_box, fill=2)
            mask_draw.line((base_x, base_y, tip_x, tip_y), fill=1, width=max(3, size // 18))
            mask_draw.ellipse((tip_x - 3, tip_y - 3, tip_x + 3, tip_y + 3), fill=1)

            frame = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
            mask = np.asarray(mask_image, dtype=np.int64).copy()
            distance = _point_to_ellipse_distance(tip_x, tip_y, organ_box)
            frames.append(frame)
            masks.append(mask)
            heatmaps.append(_tip_heatmap(size, tip_x, tip_y))
            tips.append((tip_x, tip_y))
            risks.append(float(distance <= 7.0))

        return (
            torch.tensor(np.stack(frames), dtype=torch.float32),
            torch.tensor(np.stack(masks), dtype=torch.long),
            torch.tensor(np.stack(heatmaps), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.asarray(tips), dtype=torch.float32),
            torch.tensor(np.asarray(risks), dtype=torch.float32),
        )

