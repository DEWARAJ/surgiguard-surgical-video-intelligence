#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <vector>

#include <onnxruntime_cxx_api.h>

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "Usage: surgiguard_inference <model.onnx> [iterations]\n";
    return 2;
  }
  const int iterations = argc == 3 ? std::max(1, std::atoi(argv[2])) : 100;
  constexpr std::int64_t kFrames = 4;
  constexpr std::int64_t kChannels = 3;
  constexpr std::int64_t kHeight = 160;
  constexpr std::int64_t kWidth = 288;
  const std::array<std::int64_t, 5> shape{1, kFrames, kChannels, kHeight, kWidth};
  std::vector<float> input(static_cast<std::size_t>(kFrames * kChannels * kHeight * kWidth), 0.25f);

  Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "surgiguard");
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  options.SetIntraOpNumThreads(1);
  const std::filesystem::path model_path(argv[1]);
  Ort::Session session(environment, model_path.c_str(), options);
  Ort::AllocatorWithDefaultOptions allocator;
  auto input_name = session.GetInputNameAllocated(0, allocator);
  auto output_name = session.GetOutputNameAllocated(0, allocator);
  const char* input_names[] = {input_name.get()};
  const char* output_names[] = {output_name.get()};
  auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto tensor = Ort::Value::CreateTensor<float>(memory, input.data(), input.size(), shape.data(), shape.size());

  for (int i = 0; i < 10; ++i) {
    session.Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1, output_names, 1);
  }
  std::vector<double> latencies;
  latencies.reserve(iterations);
  std::size_t output_elements = 0;
  for (int i = 0; i < iterations; ++i) {
    const auto start = std::chrono::steady_clock::now();
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1, output_names, 1);
    const auto stop = std::chrono::steady_clock::now();
    latencies.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    output_elements = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
  }
  std::sort(latencies.begin(), latencies.end());
  const double median = latencies[latencies.size() / 2];
  const double p95 = latencies[std::min(latencies.size() - 1, static_cast<std::size_t>(latencies.size() * 0.95))];
  std::cout << std::fixed << std::setprecision(3)
            << "{\"runtime\":\"onnxruntime_cpp_cpu\",\"iterations\":" << iterations
            << ",\"median_ms\":" << median << ",\"p95_ms\":" << p95
            << ",\"output_elements\":" << output_elements << "}\n";
  return 0;
}
