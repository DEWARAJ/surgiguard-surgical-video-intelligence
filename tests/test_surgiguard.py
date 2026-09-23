import sys
import unittest
from pathlib import Path

import torch
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from surgiguard.data import SyntheticSurgicalVideo  # noqa: E402
from surgiguard.cholecseg8k import WATERSHED_TO_CLASS, decode_watershed_mask  # noqa: E402
from surgiguard.losses import combined_loss  # noqa: E402
from surgiguard.metrics import class_metrics, risk_events, tip_coordinates  # noqa: E402
from surgiguard.model import SurgicalSegNet, TemporalSurgiNet  # noqa: E402


class SurgiGuardTests(unittest.TestCase):
    def test_dataset_is_deterministic(self):
        first = SyntheticSurgicalVideo(2, seed=5)[0]
        second = SyntheticSurgicalVideo(2, seed=5)[0]
        for left, right in zip(first, second):
            self.assertTrue(torch.equal(left, right))

    def test_model_output_shapes(self):
        model = TemporalSurgiNet(base_channels=4)
        video = torch.randn(2, 3, 3, 64, 64)
        segmentation, tips = model(video)
        self.assertEqual(tuple(segmentation.shape), (2, 3, 3, 64, 64))
        self.assertEqual(tuple(tips.shape), (2, 3, 1, 64, 64))

    def test_real_data_models_have_expected_shape(self):
        video = torch.rand(2, 4, 3, 64, 96)
        for temporal in (False, True):
            logits = SurgicalSegNet(base_channels=4, classes=13, temporal=temporal)(video)
            self.assertEqual(tuple(logits.shape), (2, 4, 13, 64, 96))

    def test_cholecseg8k_palette_decoding(self):
        values = list(WATERSHED_TO_CLASS)
        pixels = np.asarray(values, dtype=np.uint8).reshape(1, -1, 1).repeat(3, axis=2)
        decoded = decode_watershed_mask(Image.fromarray(pixels, mode="RGB"))
        self.assertEqual(decoded.tolist(), [[WATERSHED_TO_CLASS[value] for value in values]])

    def test_combined_loss_backpropagates(self):
        model = TemporalSurgiNet(base_channels=4)
        video, masks, tip_heatmaps, _, _ = SyntheticSurgicalVideo(1, sequence_length=3)[0]
        segmentation, tips = model(video.unsqueeze(0))
        loss, _ = combined_loss(
            segmentation,
            tips,
            masks.unsqueeze(0),
            tip_heatmaps.unsqueeze(0),
            {"cross_entropy": 1.0, "dice": 0.8, "tip": 1.5, "temporal": 0.1},
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_metrics_perfect_prediction(self):
        target = torch.tensor([[[[0, 1], [2, 0]]]])
        metrics = class_metrics(target, target)
        self.assertEqual(metrics["mean_foreground_iou"], 1.0)

    def test_risk_geometry(self):
        segmentation = torch.zeros((1, 1, 16, 16), dtype=torch.long)
        segmentation[0, 0, 8:12, 8:12] = 2
        heatmap = torch.zeros((1, 1, 1, 16, 16))
        heatmap[0, 0, 0, 7, 7] = 1.0
        event = risk_events(segmentation, tip_coordinates(heatmap), threshold=2.0)
        self.assertTrue(bool(event[0, 0]))


if __name__ == "__main__":
    unittest.main()
