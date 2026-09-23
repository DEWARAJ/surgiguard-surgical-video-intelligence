cmake_minimum_required(VERSION 3.18)
project(surgiguard_inference LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(onnxruntime CONFIG REQUIRED)
add_executable(surgiguard_inference inference.cpp)
target_link_libraries(surgiguard_inference PRIVATE onnxruntime::onnxruntime)

