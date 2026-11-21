# src/inkling/CMakeLists.txt - strata-inkling as its own project: CPU only, NVIDIA (CUDA) or AMD (HIP).
#
#   cmake -S engine/src/inkling -B build-inkling -G Ninja -DCMAKE_BUILD_TYPE=Release                  # CPU only
#   cmake -S engine/src/inkling -B build-inkling -G Ninja -DCMAKE_BUILD_TYPE=Release -DINKLING_GPU=cuda \
#         -DCMAKE_CUDA_ARCHITECTURES=native                                                           # NVIDIA
#   cmake -S engine/src/inkling -B build-inkling -G Ninja -DCMAKE_BUILD_TYPE=Release -DINKLING_GPU=hip \
#         -DGPU_TARGETS=gfx1100 (the card's gfx name; the HIP compiler the engine build uses)         # AMD
#   cmake --build build-inkling --target strata-inkling
#
# ggml comes from third_party/strata-llama (llama.cpp at the same pin as third_party/strata-ggml, with its CUDA and HIP
# backends): the GPU path is ggml's backend API, so the one program runs the dense weights on whichever backend it was
# built with. The root engine project builds the same sources without GPU backends (cmake/strata_inkling.cmake: the CPU
# tests and the CPU program).
cmake_minimum_required(VERSION 3.21)
project(strata_inkling C CXX)

set(INKLING_GPU "off" CACHE STRING "GPU backend: off, cuda or hip")
set_property(CACHE INKLING_GPU PROPERTY STRINGS off cuda hip)
option(INKLING_PORTABLE "build for other PCs (AVX2 baseline) instead of this CPU" OFF)
set(LLAMA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/strata-llama" CACHE PATH "the llama.cpp tree ggml comes from")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)           # the engine has its own thread pool
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
if(INKLING_PORTABLE)
  set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
  foreach(_f GGML_AVX GGML_AVX2 GGML_FMA GGML_F16C GGML_BMI2)
    set(${_f} ON CACHE BOOL "" FORCE)
  endforeach()
else()
  set(GGML_NATIVE ON CACHE BOOL "" FORCE)
endif()
if(INKLING_GPU STREQUAL "cuda")
  set(GGML_CUDA ON CACHE BOOL "" FORCE)
elseif(INKLING_GPU STREQUAL "hip")
  set(GGML_HIP ON CACHE BOOL "" FORCE)
elseif(NOT INKLING_GPU STREQUAL "off")
  message(FATAL_ERROR "INKLING_GPU must be off, cuda or hip (got '${INKLING_GPU}')")
endif()
add_subdirectory(${LLAMA_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)

find_package(Threads REQUIRED)
add_executable(strata-inkling main.cpp gguf.cpp file_io.cpp model.cpp cpu_ops.cpp dot_kernels.cpp gpu.cpp
               expert_store.cpp engine.cpp plan.cpp serve.cpp)
target_compile_definitions(strata-inkling PRIVATE INKLING_BACKENDS=1)
target_link_libraries(strata-inkling PRIVATE ggml ggml-base ggml-cpu Threads::Threads)
if(MSVC)
  target_sources(strata-inkling PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/strata-utf8.manifest)
else()
  target_compile_options(strata-inkling PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()
set_target_properties(strata-inkling PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
message(STATUS "strata-inkling: GPU backend ${INKLING_GPU}")
