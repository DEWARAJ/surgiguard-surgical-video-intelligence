from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


CLASS_NAMES = (
    "background", "abdominal_wall", "liver", "gastrointestinal_tract", "fat",
    "grasper", "connective_tissue", "blood", "cystic_duct",
    "l_hook_electrocautery", "gallbladder", "hepatic_vein", "liver_ligament",
)

WATERSHED_TO_CLASS = {
    0: 255, 255: 0, 50: 0, 11: 1, 21: 2, 13: 3, 12: 4, 31: 5, 23: 6,
    24: 7, 25: 8, 32: 9, 22: 10, 33: 11, 5: 12,
}

SPLIT_VIDEOS = {
    "validation": {"video17", "video52"},
    "test": {"video12", "video27"},
}


def decode_watershed_mask(image: Image.Image) -> np.ndarray:
    raw = np.asarray(image.convert("RGB"), dtype=np.uint8)[..., 0]
    decoded = np.full(raw.shape, 254, dtype=np.uint8)
    for value, class_id in WATERSHED_TO_CLASS.items():
        decoded[raw == value] = class_id
    if np.any(decoded == 254):
        unknown = np.unique(raw[decoded == 254]).tolist()
        raise ValueError(f"unknown CholecSeg8k mask values: {unknown}")
    return decoded


def _frame_number(path: Path) -> int:
    match = re.search(r"frame_(\d+)_endo\.png$", path.name)
    if not match:
        raise ValueError(f"unexpected frame name: {path.name}")
    return int(match.group(1))


def _resolve_root(root: Path) -> Path:
    nested = root / "CholecSeg8k"
    return nested if nested.is_dir() else root


class CholecSeg8kSequences(Dataset):
    """Short annotated sequences with source-video-separated splits."""

    def __init__(self, root: str | Path, split: str, image_size: tuple[int, int] = (160, 288),
                 sequence_length: int = 4, stride: int = 4, max_sequences: int | None = None,
                 seed: int = 0, augment: bool = False):
        if split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        self.root = _resolve_root(Path(root))
        self.split = split
        self.image_size = tuple(image_size)
        self.sequence_length = sequence_length
        self.augment = augment
        self.seed = seed

        all_videos = sorted(path for path in self.root.glob("video*") if path.is_dir())
        held_out = SPLIT_VIDEOS["validation"] | SPLIT_VIDEOS["test"]
        selected = [path for path in all_videos if
                    (split == "train" and path.name not in held_out) or
                    (split != "train" and path.name in SPLIT_VIDEOS[split])]
        if not selected:
            raise FileNotFoundError(f"no {split} videos found under {self.root}")

        sequences: list[tuple[Path, ...]] = []
        for video in selected:
            for clip in sorted(path for path in video.iterdir() if path.is_dir()):
                frames = sorted(clip.glob("frame_*_endo.png"), key=_frame_number)
                frames = [path for path in frames if "_mask" not in path.name]
                for start in range(0, len(frames) - sequence_length + 1, stride):
                    sequence = tuple(frames[start:start + sequence_length])
                    numbers = [_frame_number(path) for path in sequence]
                    if numbers == list(range(numbers[0], numbers[0] + sequence_length)):
                        sequences.append(sequence)
        random.Random(seed).shuffle(sequences)
        self.sequences = sequences[:max_sequences] if max_sequences else sequences
        self.video_ids = tuple(sorted(path.name for path in selected))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        height, width = self.image_size
        flip = self.augment and random.Random(self.seed + index * 7919).random() < 0.5
        frames, masks = [], []
        for image_path in self.sequences[index]:
            mask_path = image_path.with_name(image_path.name.replace("_endo.png", "_endo_watershed_mask.png"))
            image = Image.open(image_path).convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
            mask_image = Image.open(mask_path).convert("RGB").resize((width, height), Image.Resampling.NEAREST)
            if flip:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask_image = mask_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            frames.append(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)
            masks.append(decode_watershed_mask(mask_image))
        key = str(self.sequences[index][0].relative_to(self.root)).replace("\\", "/")
        return torch.tensor(np.stack(frames)), torch.tensor(np.stack(masks), dtype=torch.long), key


def split_manifest(root: str | Path, **dataset_kwargs) -> dict[str, dict[str, object]]:
    manifest = {}
    for split in ("train", "validation", "test"):
        dataset = CholecSeg8kSequences(root, split, **dataset_kwargs)
        manifest[split] = {"videos": list(dataset.video_ids), "sequences": len(dataset)}
    return manifest
