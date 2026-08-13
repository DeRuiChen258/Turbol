cmake_minimum_required(VERSION 3.28)
project(TurboRL VERSION 1.0.0 LANGUAGES CXX)

option(TURBORL_ENABLE_CUDA "Enable CUDA support" OFF)
option(TURBORL_ENABLE_TESTS "Enable testing" ON)
option(TURBORL_ENABLE_BENCHMARKS "Enable benchmarks" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-parameter")

find_package(Threads REQUIRED)

# GoogleTest
if(TURBORL_ENABLE_TESTS)
    find_package(GTest REQUIRED)
    enable_testing()
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files
file(GLOB_RECURSE CORE_SOURCES "${CMAKE_SOURCE_DIR}/src/core/*.cpp")
file(GLOB_RECURSE ENV_SOURCES "${CMAKE_SOURCE_DIR}/src/env/*.cpp")
file(GLOB_RECURSE RB_SOURCES "${CMAKE_SOURCE_DIR}/src/replay_buffer/*.cpp")
file(GLOB_RECURSE ROLLOUT_SOURCES "${CMAKE_SOURCE_DIR}/src/rollout/*.cpp")
file(GLOB_RECURSE POLICY_SOURCES "${CMAKE_SOURCE_DIR}/src/policy/*.cpp")
file(GLOB_RECURSE DIST_SOURCES "${CMAKE_SOURCE_DIR}/src/distributed/*.cpp")
file(GLOB_RECURSE UTILS_SOURCES "${CMAKE_SOURCE_DIR}/src/utils/*.cpp")

set(ALL_CPP ${CORE_SOURCES} ${ENV_SOURCES} ${RB_SOURCES} ${ROLLOUT_SOURCES} ${POLICY_SOURCES} ${DIST_SOURCES} ${UTILS_SOURCES})

add_library(turbol STATIC ${ALL_CPP})
target_link_libraries(turbol PUBLIC Threads::Threads)

# Example
add_executable(turbol_example ${CMAKE_SOURCE_DIR}/examples/main.cpp)
target_link_libraries(turbol_example PRIVATE turbol)

# Tests
if(TURBORL_ENABLE_TESTS)
    add_executable(turbol_test 
        tests/test_tensor.cpp
        tests/test_ring_buffer.cpp
        tests/test_rollout.cpp
    )
    target_link_libraries(turbol_test PRIVATE 
        turbol 
        GTest::gtest_main 
        Threads::Threads
    )
    include(GoogleTest)
    gtest_discover_tests(turbol_test)
endif()

# Benchmarks
if(TURBORL_ENABLE_BENCHMARKS)
    add_executable(benchmark_tensor benchmarks/benchmark_tensor.cpp)
    target_link_libraries(benchmark_tensor PRIVATE 
        turbol 
        Threads::Threads
    )
    
    add_executable(benchmark_attention benchmarks/benchmark_attention.cpp)
    target_link_libraries(benchmark_attention PRIVATE 
        turbol 
        Threads::Threads
    )
endif()

message(STATUS "TurboRL (Original) configured: Tests=${TURBORL_ENABLE_TESTS}, Benchmarks=${TURBORL_ENABLE_BENCHMARKS}")
