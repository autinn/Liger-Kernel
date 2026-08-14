#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include <cute/atom/copy_traits_sm100_tma.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>

#include "mlp1_fused.cuh"

namespace gate_up_swiglu_cuda {

using namespace cute;
using Element = cutlass::bfloat16_t;
using Traits1Sm = liger::Mlp1Traits<
	Element, /*TileM=*/128, /*TileN=*/128, /*TileK=*/64,
	/*Stages=*/4, /*EpiChunkN=*/64>;

#define CUDA_CHECK(expr)                                                        \
	do {                                                                          \
		cudaError_t error_ = (expr);                                                 \
		if (error_ != cudaSuccess) {                                                 \
			std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #expr, \
			             cudaGetErrorString(error_));                                  \
			std::exit(1);                                                             \
		}                                                                           \
	} while (0)

struct Shape {
	const char* name;
	int tokens;
	int hidden;
	int intermediate;
	int experts;
};

struct DeviceInputs {
	Element* x = nullptr;
	Element* b = nullptr;
	Element* c = nullptr;
	int* expert_1sm = nullptr;
	std::vector<int> host_expert_1sm;
	int m_tiles_1sm = 0;
	int padded_tokens_1sm = 0;

	DeviceInputs() = default;
	DeviceInputs(const DeviceInputs&) = delete;
	DeviceInputs& operator=(const DeviceInputs&) = delete;
	DeviceInputs(DeviceInputs&& other) noexcept
		: x(std::exchange(other.x, nullptr)),
		  b(std::exchange(other.b, nullptr)),
		  c(std::exchange(other.c, nullptr)),
		  expert_1sm(std::exchange(other.expert_1sm, nullptr)),
		  host_expert_1sm(std::move(other.host_expert_1sm)),
		  m_tiles_1sm(other.m_tiles_1sm),
		  padded_tokens_1sm(other.padded_tokens_1sm) {}
	~DeviceInputs() {
		if (x)
			cudaFree(x);
		if (b)
			cudaFree(b);
		if (c)
			cudaFree(c);
		if (expert_1sm)
			cudaFree(expert_1sm);
	}
};

struct HostInputs {
	std::vector<float> x;
	std::vector<float> b;
	std::vector<float> c;
};

struct ErrorStats {
	double mean_relative = 0.0;
	double max_relative = 0.0;
	double max_absolute = 0.0;
};

template <typename Traits>
struct OneSmFusedKernelSmem {
	liger::Mlp1FusedSmem<Traits> tile;
	typename Traits::MainloopPipelineUmma::SharedStorage pipe_storage;
};

template <typename Traits, typename TmaLoadX, typename TmaLoadW, typename TmaStoreZ>
__global__ void __launch_bounds__(Traits::NumThreads, 1)
fused_kernel(
		__grid_constant__ TmaLoadX const tma_load_x,
		__grid_constant__ TmaLoadW const tma_load_b,
		__grid_constant__ TmaLoadW const tma_load_c,
		__grid_constant__ TmaStoreZ const tma_store_z,
		const int* expert_ids,
		int num_tokens,
		int hidden_dim,
		int total_n_rows,
		int num_m_tiles,
		int num_n_tiles) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
	extern __shared__ char raw_smem[];
	auto& smem = *reinterpret_cast<OneSmFusedKernelSmem<Traits>*>(raw_smem);
	using Pipeline = typename Traits::MainloopPipelineUmma;
	using PipeState = typename Traits::PipelineState;

	int warp_id = threadIdx.x / Traits::WarpSize;
	bool is_producer = warp_id == 0;
	bool is_consumer = warp_id >= 3 && warp_id <= 11;
	int num_k_tiles = hidden_dim / Traits::TileK;

	auto pipe = liger::mlp1_make_pipe_umma<Traits>(smem.pipe_storage);

	cute::TMEM::Allocator1Sm tmem_allocator;
	constexpr int kTmemColumns = Traits::AccStages * (2 * Traits::TileN);
	if (warp_id == 3) {
		tmem_allocator.allocate(kTmemColumns, &smem.tile.tmem_base);
		__syncwarp();
	}
	__syncthreads();

	PipeState producer_state = is_producer
		? cutlass::make_producer_start_state<Pipeline>()
		: PipeState{};
	PipeState consumer_state;
	int split_idx = static_cast<int>(blockIdx.y);
	int num_splits = static_cast<int>(gridDim.y);

	for (int m = static_cast<int>(blockIdx.x); m < num_m_tiles;
	     m += static_cast<int>(gridDim.x)) {
		int expert_n_offset = expert_ids[m] * num_n_tiles;
		if (is_producer) {
			liger::mlp1_fused_producer<Traits>(
				pipe,
				producer_state,
				smem.tile,
				tma_load_x,
				tma_load_b,
				tma_load_c,
				m,
				expert_n_offset,
				num_tokens,
				hidden_dim,
				total_n_rows,
				num_n_tiles,
				num_k_tiles,
				split_idx,
				num_splits);
		} else if (is_consumer) {
			liger::mlp1_fused_consumer<Traits, 100>(
				pipe,
				consumer_state,
				smem.tile,
				tma_store_z,
				m,
				num_n_tiles * Traits::TileN,
				num_m_tiles,
				num_n_tiles,
				num_k_tiles,
				split_idx,
				num_splits);
		}
	}
	__syncthreads();
	if (warp_id == 3) {
		tmem_allocator.release_allocation_lock();
		tmem_allocator.free(smem.tile.tmem_base, kTmemColumns);
	}
#else
	__trap();
#endif
}

__global__ void fill_bf16_kernel(
		Element* output, size_t count, uint32_t seed, bool zero) {
	size_t index =
		static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
	for (size_t i = index; i < count; i += stride) {
		if (zero) {
			output[i] = Element(0.0f);
			continue;
		}
		uint32_t value =
			static_cast<uint32_t>(i) ^
			static_cast<uint32_t>(i >> 32) ^ seed;
		value ^= value >> 16;
		value *= 0x7feb352du;
		value ^= value >> 15;
		value *= 0x846ca68bu;
		value ^= value >> 16;
		float unit =
			static_cast<float>(value & 0xffffu) / 65535.0f;
		output[i] = Element((unit - 0.5f) * 0.5f);
	}
}

static int ceil_div(int value, int divisor) {
	return (value + divisor - 1) / divisor;
}

static void make_expert_map(
		const Shape& shape, DeviceInputs& inputs) {
	inputs.m_tiles_1sm = ceil_div(shape.tokens, Traits1Sm::TileM);
	inputs.padded_tokens_1sm =
		inputs.m_tiles_1sm * Traits1Sm::TileM;
	inputs.host_expert_1sm.resize(inputs.m_tiles_1sm);
	for (int m = 0; m < inputs.m_tiles_1sm; ++m) {
		int64_t row = static_cast<int64_t>(m) * Traits1Sm::TileM;
		int expert = static_cast<int>(
			row * shape.experts / std::max(1, shape.tokens));
		inputs.host_expert_1sm[m] =
			std::min(expert, shape.experts - 1);
	}
	CUDA_CHECK(cudaMalloc(
		&inputs.expert_1sm,
		inputs.host_expert_1sm.size() * sizeof(int)));
	CUDA_CHECK(cudaMemcpy(
		inputs.expert_1sm,
		inputs.host_expert_1sm.data(),
		inputs.host_expert_1sm.size() * sizeof(int),
		cudaMemcpyHostToDevice));
}

static DeviceInputs make_benchmark_inputs(const Shape& shape) {
	DeviceInputs inputs;
	make_expert_map(shape, inputs);

	size_t x_count =
		static_cast<size_t>(shape.tokens) * shape.hidden;
	size_t w_count =
		static_cast<size_t>(shape.experts) *
		shape.intermediate * shape.hidden;
	CUDA_CHECK(cudaMalloc(&inputs.x, x_count * sizeof(Element)));
	CUDA_CHECK(cudaMalloc(&inputs.b, w_count * sizeof(Element)));
	CUDA_CHECK(cudaMalloc(&inputs.c, w_count * sizeof(Element)));
	return inputs;
}

static std::vector<float> download(
		const Element* device, size_t count) {
	std::vector<Element> packed(count);
	CUDA_CHECK(cudaMemcpy(
		packed.data(),
		device,
		count * sizeof(Element),
		cudaMemcpyDeviceToHost));
	std::vector<float> output(count);
	for (size_t i = 0; i < count; ++i)
		output[i] = static_cast<float>(packed[i]);
	return output;
}

struct RefOutputs {
	std::vector<float> z;
};

static RefOutputs cpu_reference(
		const Shape& shape,
		const HostInputs& host,
		const DeviceInputs& inputs,
		bool) {
	int tokens = shape.tokens;
	int hidden = shape.hidden;
	int intermediate = shape.intermediate;
	RefOutputs output;
	output.z.assign(
		static_cast<size_t>(tokens) * intermediate, 0.0f);
	for (int token = 0; token < tokens; ++token) {
		int m_tile = token / Traits1Sm::TileM;
		int expert = inputs.host_expert_1sm[m_tile];
		const float* x_row =
			host.x.data() + static_cast<size_t>(token) * hidden;
		for (int j = 0; j < intermediate; ++j) {
			const float* b_row =
				host.b.data() +
				(static_cast<size_t>(expert) * intermediate + j) *
					hidden;
			const float* c_row =
				host.c.data() +
				(static_cast<size_t>(expert) * intermediate + j) *
					hidden;
			float gate = 0.0f;
			float up = 0.0f;
			for (int k = 0; k < hidden; ++k) {
				gate += x_row[k] * b_row[k];
				up += x_row[k] * c_row[k];
			}
			float sigmoid = 1.0f / (1.0f + std::exp(-gate));
			output.z[
				static_cast<size_t>(token) * intermediate + j] =
				gate * sigmoid * up;
		}
	}
	return output;
}

static ErrorStats compare(
		const std::vector<float>& actual,
		const std::vector<float>& expected) {
	if (actual.size() < expected.size()) {
		std::fprintf(stderr, "comparison output is too small\n");
		std::exit(2);
	}
	ErrorStats stats;
	constexpr double kAtol = 1e-3;
	for (size_t i = 0; i < expected.size(); ++i) {
		double absolute =
			std::abs(static_cast<double>(actual[i]) - expected[i]);
		double relative =
			absolute /
			std::max(std::abs(static_cast<double>(expected[i])), kAtol);
		stats.mean_relative += relative;
		stats.max_relative = std::max(stats.max_relative, relative);
		stats.max_absolute = std::max(stats.max_absolute, absolute);
	}
	stats.mean_relative /= expected.size();
	return stats;
}

template <typename Traits>
static auto make_one_sm_x_tma(Element* x, const Shape& shape) {
	auto tensor = make_tensor(
		make_gmem_ptr(x),
		make_shape(
			static_cast<int64_t>(shape.tokens),
			static_cast<int64_t>(shape.hidden)),
		make_stride(
			static_cast<int64_t>(shape.hidden), Int<1>{}));
	return make_tma_copy(
		SM90_TMA_LOAD{}, tensor, typename Traits::SmemLayoutX_1{});
}

template <typename Traits>
static auto make_one_sm_w_tma(Element* weights, const Shape& shape) {
	auto tensor = make_tensor(
		make_gmem_ptr(weights),
		make_shape(
			static_cast<int64_t>(shape.experts) *
				shape.intermediate,
			static_cast<int64_t>(shape.hidden)),
		make_stride(
			static_cast<int64_t>(shape.hidden), Int<1>{}));
	return make_tma_copy(
		SM90_TMA_LOAD{}, tensor, typename Traits::SmemLayoutW_1{});
}

template <typename Traits>
static auto make_one_sm_store_tma(
		Element* output, int padded_tokens, const Shape& shape) {
	auto tensor = make_tensor(
		make_gmem_ptr(output),
		make_shape(
			static_cast<int64_t>(padded_tokens),
			static_cast<int64_t>(shape.intermediate)),
		make_stride(
			static_cast<int64_t>(shape.intermediate), Int<1>{}));
	return make_tma_copy(
		SM90_TMA_STORE{},
		tensor,
		typename Traits::SmemLayoutStoreSlot{});
}

template <typename TmaLoadX, typename TmaLoadW, typename TmaStoreZ>
static void set_one_sm_fused_attributes(
		TmaLoadX const&, TmaLoadW const&, TmaStoreZ const&) {
	auto kernel =
		fused_kernel<Traits1Sm, TmaLoadX, TmaLoadW, TmaStoreZ>;
	CUDA_CHECK(cudaFuncSetAttribute(
		kernel,
		cudaFuncAttributeMaxDynamicSharedMemorySize,
		sizeof(OneSmFusedKernelSmem<Traits1Sm>)));
}

template <typename TmaLoadX, typename TmaLoadW, typename TmaStoreZ>
static void launch_one_sm_fused(
		TmaLoadX const& tma_x,
		TmaLoadW const& tma_b,
		TmaLoadW const& tma_c,
		TmaStoreZ const& tma_z,
		const DeviceInputs& inputs,
		const Shape& shape,
		int split,
		int grid_x,
		cudaStream_t stream) {
	auto kernel =
		fused_kernel<Traits1Sm, TmaLoadX, TmaLoadW, TmaStoreZ>;
	kernel<<<
		dim3(grid_x, split),
		Traits1Sm::NumThreads,
		sizeof(OneSmFusedKernelSmem<Traits1Sm>),
		stream>>>(
		tma_x,
		tma_b,
		tma_c,
		tma_z,
		inputs.expert_1sm,
		shape.tokens,
		shape.hidden,
		shape.experts * shape.intermediate,
		inputs.m_tiles_1sm,
		shape.intermediate / Traits1Sm::TileN);
	CUDA_CHECK(cudaGetLastError());
}

static double median(std::vector<float> values) {
	std::sort(values.begin(), values.end());
	size_t count = values.size();
	return count & 1
		? values[count / 2]
		: 0.5 * (values[count / 2 - 1] + values[count / 2]);
}

static double tflops_of(
		const Shape& shape, double milliseconds) {
	double operations =
		4.0 * static_cast<double>(shape.tokens) *
		shape.hidden * shape.intermediate;
	return operations / (milliseconds * 1e-3) / 1e12;
}

}  // namespace gate_up_swiglu_cuda
