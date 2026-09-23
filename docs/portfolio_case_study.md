# SurgiGuard: interview case study

## One-sentence pitch

I built a compact temporal vision system that segments an instrument and protected anatomy, localizes the instrument tip, estimates predictive entropy, and turns the geometry into an auditable safety alert.

## The engineering problem

Frame-by-frame segmentation can flicker and a mask alone does not answer the operational question: is the instrument tip approaching a protected region? SurgiGuard couples a ConvGRU temporal model with a geometric monitor, then evaluates segmentation, localization, temporal stability, calibration, and alert quality separately.

## What I owned

- Designed a temporal encoder-decoder with a ConvGRU bottleneck and dual segmentation/heatmap heads.
- Created a deterministic procedural video generator with moving instruments, protected anatomy, distractors, and approach/recede trajectories.
- Implemented a multi-term objective: cross entropy, soft Dice, heatmap BCE, and temporal consistency.
- Added validation-only threshold calibration for the geometric risk monitor.
- Exported the model to ONNX and checked PyTorch/ONNX numerical equivalence.
- Added unit tests, CI, a model card, and a C++ ONNX Runtime deployment scaffold.

## Evidence

On 24 held-out synthetic sequences after 10 epochs: 0.944 instrument IoU, 0.955 protected-anatomy IoU, 0.067-pixel tip error, 0.028 temporal jitter, and 0.766 risk F1. These are software-verification results on generated data, not clinical or public-dataset claims.

## Honest limitation and next experiment

The current benchmark does not prove generalization to real surgical video. The next step is a patient/video-separated evaluation on a licensed public dataset such as SAR-RARP50, EndoVis, or CholecSeg8k, followed by calibration and latency profiling on the target deployment device.

## Interview questions this project supports

1. Why use temporal memory instead of independent frames?
2. Why separate pixel metrics from safety-event metrics?
3. How would you prevent patient or neighboring-frame leakage?
4. What should happen when entropy is high or calibration is poor?
5. How would you validate ONNX parity and end-to-end latency in C++?
