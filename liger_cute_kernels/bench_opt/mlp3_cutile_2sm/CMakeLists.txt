cmake_minimum_required(VERSION 3.24)
project(mlp3_cutile_2sm_comparison LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES OFF)

find_package(CUDAToolkit REQUIRED)

if(DEFINED ENV{CUTLASS_HOME})
    set(CUTLASS_HOME "$ENV{CUTLASS_HOME}")
else()
    set(CUTLASS_HOME "/usr/local/include/cutlass")
endif()
if(NOT EXISTS "${CUTLASS_HOME}/include/cute/tensor.hpp")
    message(FATAL_ERROR
        "CUTLASS_HOME must contain include/cute/tensor.hpp (got ${CUTLASS_HOME})")
endif()

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/keep")

function(add_mlp3_cuda_target target stages)
    add_executable(${target} backends/cuda_mlp3_2sm.cu)
    target_include_directories(${target} PRIVATE
        "${CUTLASS_HOME}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../csrc/core/src/moe")
    target_link_libraries(${target} PRIVATE CUDA::cudart)
    target_compile_definitions(${target} PRIVATE
        NDEBUG
        MLP3_1SM_STAGES=4
        MLP3_2SM_TILE_N=256
        MLP3_2SM_TILE_K=64
        MLP3_2SM_STAGES=${stages}
        MLP3_2SM_EPI_N=64
        MLP3_2SM_ACC_STAGES=2
        MLP3_2SM_COMPACT_EPILOGUE=0
        MLP3_2SM_EARLY_RELEASE=0)
    target_compile_options(${target} PRIVATE
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
            --keep
            --keep-dir=${CMAKE_CURRENT_BINARY_DIR}/keep/${target}
            -Xptxas=-O3,-v,--allow-expensive-optimizations=true>)
endfunction()

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/keep/cuda_mlp3_2sm")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/keep/cuda_mlp3_2sm_s6")
add_mlp3_cuda_target(cuda_mlp3_2sm 5)
add_mlp3_cuda_target(cuda_mlp3_2sm_s6 6)
