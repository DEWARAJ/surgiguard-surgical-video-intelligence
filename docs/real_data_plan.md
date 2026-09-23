# Public-Data Validation Plan

The synthetic MVP verifies the software and deployment path. It is not evidence of surgical performance.

## Candidate datasets

### SAR-RARP50

Fifty robot-assisted radical prostatectomy suturing video segments with surgical-action recognition and instrumentation-segmentation labels. This is the strongest candidate for a multi-task temporal extension.

Paper and dataset reference: https://arxiv.org/abs/2401.00496

### EndoVis 2018 Robotic Scene Segmentation

Robotic instruments, anatomical objects, and medical devices in more complex surgical scenes than the earlier challenge sets.

Paper: https://arxiv.org/abs/2001.11190

### CholecSeg8k

8,080 pixel-annotated laparoscopic frames from 17 video clips covering 13 semantic classes.

Paper: https://arxiv.org/abs/2012.12453

## Split policy

- Split by patient/video, never by neighboring frames.
- Keep all frames from one source video in exactly one partition.
- Freeze the test partition before architecture development.
- Record file hashes and a generated split manifest.

## Experiments

1. Single-frame U-Net baseline trained from scratch.
2. SurgiGuard without temporal memory.
3. SurgiGuard with ConvGRU memory.
4. SurgiGuard with temporal consistency loss.
5. SurgiGuard with temperature-scaled confidence calibration.

## Reported metrics

- Per-class IoU and Dice.
- Instrument-tip error when annotations permit.
- Temporal mask jitter.
- Expected calibration error.
- p50 and p95 latency, throughput, GPU memory, and model size.
- Mean and standard deviation across at least three seeds.

