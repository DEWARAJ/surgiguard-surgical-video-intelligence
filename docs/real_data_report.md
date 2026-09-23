# CholecSeg8k real-data report

## Question

Does adding ConvGRU temporal memory improve segmentation and stability across unseen surgical videos compared with a matched frame-only model?

## Protocol

- Dataset: CholecSeg8k, CC BY-NC-SA 4.0.
- Resolution: 160x288.
- Sequence length: four consecutive frames.
- Training: 480 sequences sampled from 13 source videos.
- Validation: 120 sequences from video17/video52.
- Test: 120 sequences from untouched video12/video27.
- Objective: cross entropy plus soft Dice; the temporal model also receives a 0.05 temporal-consistency term.
- Selection: best validation mean foreground IoU.
- Hardware: NVIDIA RTX 4050 Laptop GPU.
- Seed: 23, reset identically before each experiment.

## Results

| Model | Validation mIoU | Test mIoU | Static-region flicker |
|---|---:|---:|---:|
| Frame-only | 0.2816 | 0.2235 | 0.02760 |
| ConvGRU | 0.2956 | 0.2086 | 0.04630 |

The ConvGRU gains 0.0140 validation mIoU but loses 0.0149 test mIoU and increases flicker by 67.8%. On this subset, temporal memory overfits the validation videos and is not the deployment choice.

## Failure interpretation

The visual failures show class confusion across tissue boundaries and instruments on the untouched videos. Rare classes are especially weak or absent in the sampled test subset. The temporal model often propagates an incorrect label across consecutive frames, explaining why temporal context can increase persistent error even when it improves validation performance.

## Decision

Export both models for reproducibility, but select the 66K-parameter frame-only model. A next temporal experiment should warp predictions with motion estimates before applying consistency and should be repeated on the full training set across several seeds.

## Evidence boundary

This is a subset benchmark with one seed. It demonstrates experimental design, leakage control, deployment, and failure analysis; it does not establish clinical performance.
