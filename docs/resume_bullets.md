# Résumé bullets

Use the first two bullets for an ML role; add the third when space permits.

- Built and trained matched PyTorch surgical-video segmentation models from scratch on CholecSeg8k using source-video-separated splits; compared a 66K-parameter frame model with a ConvGRU model across 13 classes and documented cross-video failure modes instead of selecting on validation alone.
- Developed a reproducible real-data pipeline with official mask decoding, annotation-defect handling, deterministic sequence sampling, class IoU, static-region flicker, failure visualization, automated tests, and CUDA training.
- Exported both models to ONNX with less than 1e-4 PyTorch parity error, compiled a C++17 ONNX Runtime benchmark, and measured median/p95 CPU latency across 100 post-warmup iterations.

Suggested project heading:

**SurgiGuard — Surgical Video ML & Production Inference** | PyTorch, C++17, ONNX Runtime, CUDA
