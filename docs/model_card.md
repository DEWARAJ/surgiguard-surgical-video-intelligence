# SurgiGuard Model Card

## Real-data models

The CholecSeg8k experiment compares a 66,385-parameter frame-only encoder-decoder with a 190,945-parameter ConvGRU model across 13 classes. Videos 17/52 are validation, videos 12/27 are untouched test data, and the remaining 13 videos supply training sequences. No source video crosses a split.

The temporal model improves validation mIoU from 0.282 to 0.296 but performs worse on held-out test videos (0.209 versus 0.224) and has higher static-region flicker (0.0463 versus 0.0276). The frame-only model is the deployment candidate.

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

## Training data

- Real experiment: a fixed subset of CholecSeg8k, licensed CC BY-NC-SA 4.0, with source-video-separated evaluation.
- Synthetic experiment: generated 64x64 sequences containing an instrument, protected region, distractors, tip coordinates, and proximity labels.

## Known limitations

- The real-data run uses one seed and a subset rather than every available sequence.
- Rare classes are absent from some sampled splits, making macro metrics unstable.
- CholecSeg8k represents one procedure with limited sites and devices.
- Synthetic visuals do not capture the full distribution of real surgery.
- The current geometric risk metric operates in image space and does not estimate three-dimensional distance.
- Temporal loss can penalize legitimate rapid motion because it is not motion compensated.
- Calibration measured on synthetic data does not transfer to clinical data.

## Required next validation

1. Train on every eligible sequence and repeat across at least three seeds.
2. Add confidence intervals and class-frequency-aware metrics.
3. Test motion-compensated temporal consistency.
4. Expand failure slices for smoke, blur, fluids, reflections, occlusion, and unseen instruments.
5. Measure complete camera-to-overlay latency on the target robotic computer.
