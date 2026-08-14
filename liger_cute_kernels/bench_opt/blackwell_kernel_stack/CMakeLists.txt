cmake_minimum_required(VERSION 3.24)
project(blackwell_kernel_stack LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES OFF)

find_package(CUDAToolkit REQUIRED)
find_package(GTest REQUIRED)

if(DEFINED ENV{CUTLASS_HOME})
    set(CUTLASS_HOME "$ENV{CUTLASS_HOME}")
else()
    set(CUTLASS_HOME "/usr/local/include/cutlass")
endif()
if(NOT EXISTS "${CUTLASS_HOME}/include/cute/tensor.hpp")
    message(FATAL_ERROR
        "CUTLASS_HOME must contain include/cute/tensor.hpp (got ${CUTLASS_HOME})")
endif()

add_executable(test_mlp1_fused_2sm test_mlp1_fused_2sm.cu)
target_include_directories(test_mlp1_fused_2sm PRIVATE
    "${CUTLASS_HOME}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/backends"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../csrc/core/src/moe")
target_link_libraries(test_mlp1_fused_2sm PRIVATE
    GTest::gtest
    CUDA::cudart)
target_compile_options(test_mlp1_fused_2sm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -gencode=arch=compute_100a,code=sm_100a
        --use_fast_math
        --extra-device-vectorization
        --fmad=true
        --prec-div=false
        --prec-sqrt=false
        --generate-line-info
        --expt-relaxed-constexpr
        --expt-extended-lambda
        -Xptxas=-O3,-v,--allow-expensive-optimizations=true>)
