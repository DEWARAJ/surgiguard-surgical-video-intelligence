from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from surgiguard.cholecseg8k import CLASS_NAMES  # noqa: E402
from surgiguard.model import SurgicalSegNet  # noqa: E402


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * fraction), len(ordered) - 1)]


def export_one(name: str, temporal: bool) -> dict[str, object]:
    checkpoint = torch.load(ROOT / "artifacts" / f"cholecseg8k_{name}.pt", map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    model = SurgicalSegNet(config["base_channels"], len(CLASS_NAMES), temporal=temporal)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    height, width = config["image_size"]
    sample = torch.rand(1, config["sequence_length"], 3, height, width)
    output = ROOT / "artifacts" / f"cholecseg8k_{name}.onnx"
    torch.onnx.export(
        model, sample, output, input_names=["video"], output_names=["segmentation_logits"],
        opset_version=17, dynamo=False,
    )
    with torch.no_grad():
        expected = model(sample).numpy()
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"video": sample.numpy()})[0]
    for _ in range(10):
        session.run(None, {"video": sample.numpy()})
    latencies = []
    for _ in range(50):
        start = time.perf_counter()
        session.run(None, {"video": sample.numpy()})
        latencies.append((time.perf_counter() - start) * 1000.0)
    return {
        "model": name,
        "onnx_file": output.name,
        "file_bytes": output.stat().st_size,
        "maximum_absolute_error": float(np.max(np.abs(expected - actual))),
        "equivalent_at_1e-4": bool(np.allclose(expected, actual, atol=1e-4, rtol=1e-4)),
        "cpu_onnxruntime_latency_ms": {
            "median": statistics.median(latencies),
            "p95": percentile(latencies, 0.95),
            "iterations": len(latencies),
        },
    }


def main() -> None:
    report = {
        "evidence_type": "real_cholecseg8k_subset",
        "input_shape": [1, 4, 3, 160, 288],
        "exports": [export_one("frame_only", False), export_one("temporal", True)],
    }
    (ROOT / "results" / "cholecseg8k_onnx.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
