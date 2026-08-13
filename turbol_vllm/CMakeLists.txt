cmake_minimum_required(VERSION 3.28)
project(TurboRL VERSION 1.0.0 LANGUAGES C CXX CUDA)

# ============================================================================
# Options
# ============================================================================
option(TURBORL_ENABLE_CUDA    "Enable CUDA support"             ON)
option(TURBORL_USE_NCCL       "Enable NCCL distributed support" OFF)
option(TURBORL_USE_LIBTORCH   "Enable libtorch support"         ON)
option(TURBORL_ENABLE_TESTS   "Enable testing"                 ON)
option(TURBORL_ENABLE_BENCHMARKS "Enable benchmarks"           ON)

# ============================================================================
# Language standards
#   Host code   : C++23
#   C code      : C17
#   CUDA device : C++20 (CUDA 13.2 maximum stable device standard)
# ============================================================================
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# ============================================================================
# CUDA 13.2 settings — target RTX 5070 (Blackwell, sm_120)
# ============================================================================
if(TURBORL_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    # sm_120 = Blackwell (RTX 5070). Provide a PTX fallback for forward compat.
    set(CMAKE_CUDA_ARCHITECTURES 120)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --use_fast_math")
    find_package(CUDAToolkit REQUIRED)
    add_definitions(-DTURBORL_CUDA=1)
endif()

# ============================================================================
# libtorch (PyTorch C++ frontend) — CUDA 13.2 build
# ============================================================================
if(TURBORL_USE_LIBTORCH)
    set(LIBTORCH_ROOT "/opt/libtorch"
        CACHE PATH "Path to libtorch distribution")
    # libtorch ignores CMAKE_CUDA_ARCHITECTURES; honor its own variable instead.
    set(TORCH_CUDA_ARCH_LIST "12.0")
    list(PREPEND CMAKE_PREFIX_PATH "${LIBTORCH_ROOT}")
    find_package(Torch REQUIRED)
    add_definitions(-DTURBORL_LIBTORCH=1)
    message(STATUS "libtorch: ENABLED at ${LIBTORCH_ROOT}")
endif()

# ============================================================================
# NCCL (optional — distributed collectives)
# ============================================================================
if(TURBORL_USE_NCCL AND TURBORL_ENABLE_CUDA)
    find_path(NCCL_INCLUDE_DIR nccl.h PATHS /usr/include NO_DEFAULT_PATH)
    find_library(NCCL_LIBRARY nccl PATHS /usr/lib/x86_64-linux-gnu NO_DEFAULT_PATH)
    if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
        add_definitions(-DTURBORL_NCCL_ENABLED=1)
        message(STATUS "NCCL: ENABLED")
    else()
        message(WARNING "NCCL requested but not found — disabling")
        set(TURBORL_USE_NCCL OFF)
    endif()
endif()

# ============================================================================
# vLLM inference backend (sibling C++ inference engine)
# ============================================================================
set(VLLM_ROOT "${CMAKE_SOURCE_DIR}/../../vllm")
set(VLLM_BUILD "${VLLM_ROOT}/build")
if(EXISTS "${VLLM_BUILD}/libvllm.a")
    set(VLLM_INCLUDE_DIR "${VLLM_ROOT}/include")
    set(VLLM_LIBRARY "${VLLM_BUILD}/libvllm.a")
    add_definitions(-DTURBORL_VLLM_ENABLED=1)
    message(STATUS "vLLM: ENABLED (real library)")
endif()

find_package(Threads REQUIRED)
if(TURBORL_ENABLE_TESTS)
    find_package(GTest REQUIRED)
    enable_testing()
endif()

# ============================================================================
# Sources (explicit — deterministic build graph)
# ============================================================================
set(CORE_SOURCES
    src/core/core_engine.cpp
    src/core/config.cpp)
set(ENV_SOURCES
    src/env/gpu_environment.cpp)
set(RB_SOURCES
    src/replay_buffer/replay_buffer.cpp)
set(ROLLOUT_SOURCES
    src/rollout/rollout_engine.cpp
    src/rollout/vllm_backend.cpp)
set(POLICY_SOURCES
    src/policy/policy_engine.cpp)
set(DIST_SOURCES
    src/distributed/distributed_trainer.cpp
    src/distributed/nccl_communicator.cpp)
set(UTILS_SOURCES
    src/utils/tensor_utils.cpp)
set(PROFILER_SOURCES
    src/profiler/profiler.cpp
    src/profiler/tensorboard_writer.cpp)
set(REWARD_SOURCES
    src/reward/reward_engine.cpp)
set(DATASET_SOURCES
    src/dataset/dataset.cpp)
set(PIPELINE_SOURCES
    src/distributed/pipeline_trainer.cpp)
set(CUDA_SOURCES
    src/cuda/attention.cu
    src/cuda/ring_buffer.cu
    src/cuda/math_utils.cu
    src/cuda/env_dynamics.cu
    src/cuda/gae.cu)

set(ALL_CPP ${CORE_SOURCES} ${ENV_SOURCES} ${RB_SOURCES}
            ${ROLLOUT_SOURCES} ${POLICY_SOURCES} ${DIST_SOURCES} ${UTILS_SOURCES}
            ${PROFILER_SOURCES} ${REWARD_SOURCES} ${DATASET_SOURCES} ${PIPELINE_SOURCES})

# ============================================================================
# Library
# ============================================================================
add_library(turbol STATIC ${ALL_CPP})
target_include_directories(turbol PUBLIC ${CMAKE_SOURCE_DIR}/include)

if(TURBORL_ENABLE_CUDA)
    add_library(turbol_cuda OBJECT ${CUDA_SOURCES})
    target_link_libraries(turbol_cuda PRIVATE CUDA::cudart)
    target_sources(turbol PRIVATE $<TARGET_OBJECTS:turbol_cuda>)
    target_include_directories(turbol PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
    # --no-as-needed around cublas/cublasLt: libtorch.so references their symbols
    # but is linked *after* them, so the default --as-needed would drop them and
    # the final link fails with "DSO missing from command line".
    target_link_libraries(turbol PUBLIC
        CUDA::cudart
        "-Wl,--no-as-needed" CUDA::cublas CUDA::cublasLt "-Wl,--as-needed")
endif()

if(TURBORL_USE_LIBTORCH)
    target_include_directories(turbol PUBLIC ${TORCH_INCLUDE_DIRS})
    target_link_libraries(turbol PUBLIC ${TORCH_LIBRARIES})
endif()
if(TURBORL_USE_NCCL)
    target_include_directories(turbol PUBLIC ${NCCL_INCLUDE_DIR})
    target_link_libraries(turbol PUBLIC ${NCCL_LIBRARY})
endif()
if(DEFINED VLLM_LIBRARY)
    target_include_directories(turbol PUBLIC ${VLLM_INCLUDE_DIR})
    target_link_libraries(turbol PUBLIC ${VLLM_LIBRARY})
endif()

target_link_libraries(turbol PUBLIC Threads::Threads)

# ============================================================================
# Executables
# ============================================================================
add_executable(turbol_example examples/main.cpp)
target_link_libraries(turbol_example PRIVATE turbol)

if(TURBORL_ENABLE_TESTS)
    add_executable(turbol_test
        tests/test_tensor.cpp
        tests/test_ring_buffer.cpp
        tests/test_rollout.cpp
        tests/test_config.cpp
        tests/test_tensor_utils.cpp
        tests/test_profiler_reward.cpp)
    target_link_libraries(turbol_test PRIVATE turbol GTest::gtest_main Threads::Threads)
    include(GoogleTest)
    gtest_discover_tests(turbol_test)
endif()

if(TURBORL_ENABLE_BENCHMARKS)
    add_executable(benchmark_tensor benchmarks/benchmark_tensor.cpp)
    target_link_libraries(benchmark_tensor PRIVATE turbol)
    add_executable(benchmark_attention benchmarks/benchmark_attention.cpp)
    target_link_libraries(benchmark_attention PRIVATE turbol)
endif()

# ============================================================================
# Python bindings (pybind11)
# ============================================================================
find_package(pybind11 CONFIG QUIET)
if(pybind11_FOUND)
    message(STATUS "pybind11: FOUND — building Python bindings")
    add_subdirectory(python)
else()
    message(STATUS "pybind11: NOT FOUND — Python bindings disabled")
    message(STATUS "  Install with: pip install pybind11")
endif()

message(STATUS "========================================")
message(STATUS "TurboRL Configuration:")
message(STATUS "  C++ standard : C++23")
message(STATUS "  C standard   : C17")
message(STATUS "  CUDA         : ${TURBORL_ENABLE_CUDA} (sm_120, CUDA 13.2)")
message(STATUS "  NCCL         : ${TURBORL_USE_NCCL}")
message(STATUS "  vLLM         : ${VLLM_LIBRARY}")
message(STATUS "  libtorch     : ${TURBORL_USE_LIBTORCH}")
message(STATUS "  Tests        : ${TURBORL_ENABLE_TESTS}")
message(STATUS "========================================")
