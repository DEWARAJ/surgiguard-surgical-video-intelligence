// Minimal ONNX Runtime deployment scaffold. Image decoding, normalization,
// overlay rendering, and latency measurement belong in the application layer.
#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: surgiguard_inference <model.onnx>\n";
    return 2;
  }
  constexpr std::int64_t kFrames = 4;
  constexpr std::int64_t kChannels = 3;
  constexpr std::int64_t kHeight = 64;
  constexpr std::int64_t kWidth = 64;
  const std::array<std::int64_t, 5> shape{1, kFrames, kChannels, kHeight, kWidth};
  std::vector<float> input(static_cast<std::size_t>(kFrames * kChannels * kHeight * kWidth), 0.0f);

  Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "surgiguard");
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  Ort::Session session(environment, argv[1], options);
  Ort::AllocatorWithDefaultOptions allocator;
  auto input_name = session.GetInputNameAllocated(0, allocator);
  const char* input_names[] = {input_name.get()};
  const char* output_names[] = {"segmentation_logits", "tip_heatmaps"};
  auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto tensor = Ort::Value::CreateTensor<float>(
      memory, input.data(), input.size(), shape.data(), shape.size());
  auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1, output_names, 2);
  std::cout << "Loaded model and produced " << outputs.size() << " outputs\n";
  return 0;
}

