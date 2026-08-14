cmake_minimum_required(VERSION 3.24)
project(gate_up_swiglu_comparison LANGUAGES CXX CUDA)

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
	message(FATAL_ERROR "CUTLASS_HOME must contain include/cute/tensor.hpp (got ${CUTLASS_HOME})")
endif()

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/keep")

add_executable(cuda_cpp backends/cuda_cpp.cu)
target_include_directories(cuda_cpp PRIVATE
	"${CUTLASS_HOME}/include"
	"${CMAKE_CURRENT_SOURCE_DIR}/../../csrc/core/src/moe")
target_link_libraries(cuda_cpp PRIVATE CUDA::cudart)
target_compile_definitions(cuda_cpp PRIVATE NDEBUG)
target_compile_options(cuda_cpp PRIVATE
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
		--keep-dir=${CMAKE_CURRENT_BINARY_DIR}/keep
		-Xptxas=-O3,-v,--allow-expensive-optimizations=true>)
