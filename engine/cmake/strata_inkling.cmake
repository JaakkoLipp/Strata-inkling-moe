# cmake/strata_inkling.cmake - strata-inkling, the engine for Inkling 2 (thinkingmachines/Inkling-2-Mini GGUFs): the
# whole forward pass on the CPU through ggml-cpu's per-type dot products, the experts in RAM or streamed from the SSD
# through a fixed RAM budget (src/inkling, docs/INKLING.md). It needs no GPU toolchain, so every build kind can have it.
# This is the CPU program and its tests; the NVIDIA and AMD builds come from src/inkling/CMakeLists.txt (ggml's CUDA / HIP
# backends from third_party/strata-llama, which this project does not build).
#
# Like the other parts of this tree the block is a macro, run by CMakeLists.txt in its own scope.

option(STRATA_BUILD_INKLING "Build strata-inkling, the Inkling 2 engine (CPU, experts streamed from the SSD)" ON)

macro(strata_inkling)
  if(STRATA_BUILD_INKLING)
    if(NOT TARGET ggml-cpu)
      # a release configure links no ggml otherwise (CMakeLists.txt: own CPU pool, tests off): this engine needs ggml-cpu
      enable_language(C)
      strata_ggml_resolve("${CMAKE_CURRENT_SOURCE_DIR}")
      strata_ggml_configure_root()
    endif()
    find_package(Threads REQUIRED)
    add_library(strata_inkling STATIC
      src/inkling/gguf.cpp
      src/inkling/file_io.cpp
      src/inkling/model.cpp
      src/inkling/cpu_ops.cpp
      src/inkling/dot_kernels.cpp
      src/inkling/gpu.cpp
      src/inkling/expert_store.cpp
      src/inkling/engine.cpp
      src/inkling/plan.cpp
      src/inkling/serve.cpp)
    target_include_directories(strata_inkling PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/inkling)
    target_link_libraries(strata_inkling PUBLIC ggml-cpu ggml-base Threads::Threads)
    if(NOT MSVC)
      target_compile_options(strata_inkling PRIVATE -Wall -Wextra -Wno-unused-parameter)
    endif()
    add_executable(strata-inkling src/inkling/main.cpp)
    target_link_libraries(strata-inkling PRIVATE strata_inkling)
    strata_utf8_manifest(strata-inkling)
    install(TARGETS strata-inkling RUNTIME DESTINATION bin)

    if(STRATA_BUILD_TESTS)
      # no model and no GPU: the slot cache, the GGUF reader and the forward pass on a tiny synthetic model
      add_executable(inkling_store_test src/inkling/tests/store_test.cpp)
      target_link_libraries(inkling_store_test PRIVATE strata_inkling)
      add_test(NAME inkling_store_test COMMAND inkling_store_test)
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/inkling/tests/tiny_model_test.cpp)
        add_executable(inkling_tiny_test src/inkling/tests/tiny_model_test.cpp)
        target_link_libraries(inkling_tiny_test PRIVATE strata_inkling)
        add_test(NAME inkling_tiny_test COMMAND inkling_tiny_test)
      endif()
    endif()
  endif()
endmacro()
