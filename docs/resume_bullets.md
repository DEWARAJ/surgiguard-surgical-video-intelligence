# Résumé bullets

Use these only after you can explain the implementation and the evidence boundary.

- Built a custom PyTorch ConvGRU video model from scratch to segment instruments/protected anatomy and localize instrument tips; reached 0.950 mean foreground IoU and 0.067-pixel tip error on 24 held-out synthetic sequences.
- Designed a geometric safety monitor with validation-only threshold calibration, predictive entropy, temporal-jitter measurement, and risk-event precision/recall/F1 reporting.
- Exported the 85K-parameter model to ONNX, verified PyTorch/ONNX Runtime outputs within 1e-4, and added a C++ ONNX Runtime deployment scaffold plus automated tests and CI.

Suggested project heading:

**SurgiGuard — Uncertainty-Aware Surgical Video Intelligence** | PyTorch, C++, ONNX Runtime, temporal vision
