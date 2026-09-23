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
- Added unit tests, CI, a model card, real-data failure analysis, and a compiled C++ ONNX Runtime benchmark.

## Evidence

On real CholecSeg8k data, I compared matched frame-only and ConvGRU models with source-video-separated splits. The temporal model improved validation mIoU but fell from 0.224 to 0.209 on untouched test videos and increased static-region flicker, so I selected the smaller frame model. The synthetic pipeline remains separately labeled as software-verification evidence.

## Honest limitation and next experiment

The real-data run uses a fixed subset and one seed, so it does not establish clinical generalization. The next step is full-dataset multi-seed evaluation, motion-compensated temporal consistency, and end-to-end latency profiling on the target deployment device.

## Interview questions this project supports

1. Why use temporal memory instead of independent frames?
2. Why separate pixel metrics from safety-event metrics?
3. How would you prevent patient or neighboring-frame leakage?
4. What should happen when entropy is high or calibration is poor?
5. How would you validate ONNX parity and end-to-end latency in C++?
