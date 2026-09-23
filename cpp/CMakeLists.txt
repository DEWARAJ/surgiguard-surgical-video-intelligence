cmake_minimum_required(VERSION 3.18)
project(surgiguard_inference LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ORT_HEADER_DIR "" CACHE PATH "ONNX Runtime include directory")
set(ORT_LIBRARY_DIR "" CACHE PATH "ONNX Runtime library directory")
if(NOT EXISTS "${ORT_HEADER_DIR}/onnxruntime_cxx_api.h")
  message(FATAL_ERROR "Set ORT_HEADER_DIR to the ONNX Runtime include directory")
endif()
find_library(ONNXRUNTIME_LIBRARY NAMES onnxruntime PATHS "${ORT_LIBRARY_DIR}" NO_DEFAULT_PATH REQUIRED)

add_executable(surgiguard_inference inference.cpp)
target_include_directories(surgiguard_inference PRIVATE "${ORT_HEADER_DIR}")
target_link_libraries(surgiguard_inference PRIVATE "${ONNXRUNTIME_LIBRARY}")
if(WIN32)
  add_custom_command(TARGET surgiguard_inference POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${ORT_LIBRARY_DIR}/onnxruntime.dll" "$<TARGET_FILE_DIR:surgiguard_inference>")
endif()
