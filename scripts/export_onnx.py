from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from surgiguard.model import TemporalSurgiNet  # noqa: E402


def main() -> None:
    checkpoint_path = ROOT / "artifacts" / "surgiguard_synthetic.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = TemporalSurgiNet(config["base_channels"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    sample = torch.randn(1, config["sequence_length"], 3, config["image_size"], config["image_size"])
    output_path = ROOT / "artifacts" / "surgiguard_synthetic.onnx"
    torch.onnx.export(
        model,
        sample,
        output_path,
        input_names=["video"],
        output_names=["segmentation_logits", "tip_heatmaps"],
        dynamic_axes={"video": {0: "batch"}, "segmentation_logits": {0: "batch"}, "tip_heatmaps": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(output_path))
    with torch.no_grad():
        expected = model(sample)
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"video": sample.numpy()})
    maximum_errors = [float(np.max(np.abs(reference.numpy() - candidate))) for reference, candidate in zip(expected, actual)]
    report = {
        "evidence_type": "synthetic_only",
        "onnx_file": output_path.name,
        "maximum_absolute_errors": maximum_errors,
        "equivalent_at_1e-4": all(error < 1e-4 for error in maximum_errors),
    }
    (ROOT / "results" / "onnx_equivalence.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

