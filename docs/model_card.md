# SurgiGuard Model Card

## Model

`TemporalSurgiNet` is a compact encoder-decoder with ConvGRU temporal memory and two output heads:

- Three-class semantic segmentation: background/tissue, instrument, protected anatomy.
- Instrument-tip probability heatmap.

The geometric safety monitor calculates the minimum image-plane distance between the predicted tip and protected-anatomy pixels.

## Intended use

- Portfolio demonstration of surgical ML engineering.
- Software architecture and reproducibility research.
- Controlled benchmarking on appropriately licensed public datasets.

## Excluded use

- Clinical decision making.
- Diagnosis or treatment.
- Autonomous control of a medical robot.
- Claims about patient or clinical performance.

## Current training data

Procedurally generated 64x64 video sequences containing a moving instrument, a protected anatomical region, tissue-like texture, specular distractors, segmentation labels, instrument-tip coordinates, and proximity labels.

## Known limitations

- Synthetic visuals do not capture real surgical anatomy, smoke, fluids, lens contamination, or device diversity.
- The current geometric risk metric operates in image space and does not estimate three-dimensional distance.
- Temporal loss can penalize legitimate rapid motion because it is not motion compensated.
- Calibration measured on synthetic data does not transfer to clinical data.

## Required next validation

1. Video/patient-separated evaluation on a public surgical dataset.
2. Comparison against a documented U-Net baseline.
3. Multiple training seeds with confidence intervals.
4. Ablation of ConvGRU memory, temporal loss, and tip head.
5. Failure slices for occlusion, smoke, motion blur, specular highlights, rare instruments, and rapid camera motion.
6. ONNX Runtime C++ latency measurement on target hardware.

