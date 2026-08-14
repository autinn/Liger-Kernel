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

#include "mlp3.cuh"
#include "mlp3_2sm.cuh"

using namespace cute;

namespace bench_mlp3_2sm {

using Element = cutlass::bfloat16_t;

#ifndef MLP3_1SM_STAGES
#define MLP3_1SM_STAGES 2
#endif

using Traits1Sm = liger::Mlp3Traits<
	Element, 128, 256, 64, MLP3_1SM_STAGES, 64>;

#ifndef MLP3_2SM_TILE_N
#define MLP3_2SM_TILE_N 256
#endif
#ifndef MLP3_2SM_TILE_K
#define MLP3_2SM_TILE_K 64
#endif
#ifndef MLP3_2SM_STAGES
#define MLP3_2SM_STAGES 3
#endif
#ifndef MLP3_2SM_EPI_N
#define MLP3_2SM_EPI_N 64
#endif
#ifndef MLP3_2SM_ACC_STAGES
#define MLP3_2SM_ACC_STAGES 2
#endif
#ifndef MLP3_2SM_COMPACT_EPILOGUE
#define MLP3_2SM_COMPACT_EPILOGUE 0
#endif
// mlp3_s5_early_release experiment (default off — see EarlyTmemRelease_ on
// Mlp3Traits2Sm in mlp3_2sm.cuh). Independent knob from
// MLP3_2SM_COMPACT_EPILOGUE; this experiment only varies it holding
// MLP3_2SM_STAGES=5, MLP3_2SM_COMPACT_EPILOGUE=0.
#ifndef MLP3_2SM_EARLY_RELEASE
#define MLP3_2SM_EARLY_RELEASE 0
#endif

using Traits2Sm = liger::Mlp3Traits2Sm<
	Element, 256, MLP3_2SM_TILE_N, MLP3_2SM_TILE_K, MLP3_2SM_STAGES,
	MLP3_2SM_EPI_N, MLP3_2SM_ACC_STAGES, 2,
	(MLP3_2SM_COMPACT_EPILOGUE != 0),
	(MLP3_2SM_EARLY_RELEASE != 0)>;

constexpr size_t kL2EvictionBytes = 256ull << 20;

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
	Element* dy = nullptr;
	Element* z = nullptr;
	int* k_starts = nullptr;
	int* k_ends = nullptr;
	int* k_starts_2sm = nullptr;
	int* k_ends_2sm = nullptr;

	DeviceInputs() = default;
	DeviceInputs(const DeviceInputs&) = delete;
	DeviceInputs& operator=(const DeviceInputs&) = delete;

	DeviceInputs(DeviceInputs&& other) noexcept {
		dy = std::exchange(other.dy, nullptr);
		z = std::exchange(other.z, nullptr);
		k_starts = std::exchange(other.k_starts, nullptr);
		k_ends = std::exchange(other.k_ends, nullptr);
		k_starts_2sm = std::exchange(other.k_starts_2sm, nullptr);
		k_ends_2sm = std::exchange(other.k_ends_2sm, nullptr);
	}

	~DeviceInputs() {
		if (dy) cudaFree(dy);
		if (z) cudaFree(z);
		if (k_starts) cudaFree(k_starts);
		if (k_ends) cudaFree(k_ends);
		if (k_starts_2sm) cudaFree(k_starts_2sm);
		if (k_ends_2sm) cudaFree(k_ends_2sm);
	}
};

struct BenchmarkResult {
	double ms = 0.0;
	double tflops = 0.0;
	int split = 1;
	int grid = 1;
};

template <typename Traits>
struct Mlp3OneSmKernelSmem {
	liger::Mlp3Smem<Traits> tile;
	typename Traits::MainloopPipelineUmma::SharedStorage pipe_storage;
};

template <
	typename Traits,
	typename TmaLoadDYT,
	typename TmaLoadZ,
	typename TmaReduceDA>
__global__ void __launch_bounds__(Traits::NumThreads, 1)
mlp3_one_sm_kernel(
		__grid_constant__ TmaLoadDYT const tma_load_dyt,
		__grid_constant__ TmaLoadZ const tma_load_z,
		__grid_constant__ TmaReduceDA const tma_reduce_da,
		const int* expert_k_starts,
		const int* expert_k_ends,
		int num_experts,
		int hidden_dim,
		int intermediate_dim,
		int num_tokens,
		int total_n_rows,
		int num_m_tiles,
		int num_n_tiles,
		int outer_split,
		int k_split = 1) {
	extern __shared__ char raw_smem[];
	auto& smem =
		*reinterpret_cast<Mlp3OneSmKernelSmem<Traits>*>(raw_smem);

	using Pipeline = typename Traits::MainloopPipelineUmma;
	using PipeState = typename Traits::PipelineState;

	int warp_id = threadIdx.x / Traits::WarpSize;
	bool is_producer = warp_id == 0;
	bool is_consumer = warp_id >= 3 && warp_id <= 11;
	auto pipe = liger::mlp3_make_pipe_umma<Traits>(smem.pipe_storage);

	cute::TMEM::Allocator1Sm tmem_allocator;
	constexpr int kTmemColumns = Traits::AccStages * Traits::TileN;
	if (warp_id == 3) {
		tmem_allocator.allocate(kTmemColumns, &smem.tile.tmem_base);
		__syncwarp();
	}
	__syncthreads();

	PipeState producer_state = is_producer
		? cutlass::make_producer_start_state<Pipeline>()
		: PipeState{};
	PipeState consumer_state;

	int cell_start = static_cast<int>(blockIdx.x);
	int cell_stride = static_cast<int>(gridDim.x);
	int batch_kb_end = num_tokens / Traits::TileK;

	if (is_producer) {
		liger::mlp3_producer<Traits>(
			pipe, producer_state, smem.tile,
			tma_load_dyt, tma_load_z,
			expert_k_starts, expert_k_ends, num_experts,
			hidden_dim, intermediate_dim, num_tokens,
			num_m_tiles, num_n_tiles, outer_split,
			cell_start, cell_stride,
			/*batch_kb_start=*/0, batch_kb_end, k_split);
	} else if (is_consumer) {
		liger::mlp3_consumer<Traits, 100>(
			pipe, consumer_state, smem.tile, tma_reduce_da,
			expert_k_starts, expert_k_ends, num_experts,
			intermediate_dim, total_n_rows,
			num_m_tiles, num_n_tiles, outer_split,
			cell_start, cell_stride,
			/*batch_kb_start=*/0, batch_kb_end, k_split);
	}

	__syncthreads();
	if (warp_id == 3) {
		tmem_allocator.release_allocation_lock();
		tmem_allocator.free(smem.tile.tmem_base, kTmemColumns);
	}
}

template <
	typename Traits,
	typename TmaLoadDYT,
	typename TmaLoadZ,
	typename TmaReduceDA>
__global__ void __launch_bounds__(Traits::NumThreads, 1)
mlp3_two_sm_kernel(
		__grid_constant__ TmaLoadDYT const tma_load_dyt,
		__grid_constant__ TmaLoadZ const tma_load_z,
		__grid_constant__ TmaReduceDA const tma_reduce_da,
		const int* expert_k_starts,
		const int* expert_k_ends,
		int num_experts,
		int hidden_dim,
		int intermediate_dim,
		int num_tokens,
		int num_m_tiles,
		int num_n_tiles,
		int outer_split,
		int k_split = 1) {
	extern __shared__ char raw_smem[];
	auto& smem =
		*reinterpret_cast<liger::Mlp3Smem2Sm<Traits>*>(raw_smem);
	liger::mlp3_fwd_2sm<Traits>(
		smem, tma_load_dyt, tma_load_z, tma_reduce_da,
		expert_k_starts, expert_k_ends, num_experts,
		hidden_dim, intermediate_dim, num_tokens,
		num_m_tiles, num_n_tiles, outer_split,
		/*batch_kb_start=*/0, /*batch_kb_end=*/-1,
		k_split, /*ring_kb=*/0);
}

__global__ void fill_bf16_kernel(
		Element* output, size_t count, uint32_t seed) {
	size_t index =
		static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
	for (size_t i = index; i < count; i += stride) {
		uint32_t x = static_cast<uint32_t>(i) ^ seed;
		x ^= x >> 16;
		x *= 0x7feb352du;
		x ^= x >> 15;
		x *= 0x846ca68bu;
		x ^= x >> 16;
		float value =
			(static_cast<float>(x & 0xffffu) / 32768.0f - 1.0f) * 0.5f;
		output[i] = Element(value);
	}
}

__global__ void evict_l2_kernel(
		uint32_t* buffer, size_t count, uint32_t salt) {
	size_t index =
		static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
	for (size_t i = index; i < count; i += stride)
		buffer[i] = static_cast<uint32_t>(i) ^ salt;
}

static int sm_count() {
	int device = 0;
	cudaDeviceProp properties{};
	CUDA_CHECK(cudaGetDevice(&device));
	CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
	return properties.multiProcessorCount;
}

static std::vector<int> divisors(int value) {
	std::vector<int> result;
	for (int divisor = 1; divisor <= value; ++divisor)
		if (value % divisor == 0) result.push_back(divisor);
	return result;
}

static int env_int(const char* name, int fallback) {
	const char* value = std::getenv(name);
	return value == nullptr ? fallback : std::atoi(value);
}

static std::vector<int> split_candidates(
		int value, const char* fixed_split_env) {
	int fixed = env_int(fixed_split_env, 0);
	if (fixed > 0 && value % fixed != 0) {
		std::fprintf(
			stderr, "%s=%d must divide %d\n",
			fixed_split_env, fixed, value);
		std::exit(4);
	}
	return fixed > 0 ? std::vector<int>{fixed} : divisors(value);
}

static double median(std::vector<float> values) {
	std::sort(values.begin(), values.end());
	size_t size = values.size();
	return size & 1
		? values[size / 2]
		: 0.5 * (values[size / 2 - 1] + values[size / 2]);
}

static double tflops(const Shape& shape, double milliseconds) {
	double operations =
		2.0 * shape.tokens * shape.hidden * shape.intermediate;
	return operations / (milliseconds * 1e-3) / 1e12;
}

static DeviceInputs make_inputs(const Shape& shape) {
	DeviceInputs inputs;
	size_t dy_count = static_cast<size_t>(shape.tokens) * shape.hidden;
	size_t z_count =
		static_cast<size_t>(shape.tokens) * shape.intermediate;
	CUDA_CHECK(cudaMalloc(&inputs.dy, dy_count * sizeof(Element)));
	CUDA_CHECK(cudaMalloc(&inputs.z, z_count * sizeof(Element)));

	fill_bf16_kernel<<<1024, 256>>>(inputs.dy, dy_count, 0x12345678u);
	fill_bf16_kernel<<<1024, 256>>>(inputs.z, z_count, 0x9abcdef0u);
	CUDA_CHECK(cudaGetLastError());

	int k_blocks = shape.tokens / Traits1Sm::TileK;
	if (k_blocks % shape.experts != 0) {
		std::fprintf(
			stderr, "%s: token K-blocks do not divide across experts\n",
			shape.name);
		std::exit(1);
	}
	int blocks_per_expert = k_blocks / shape.experts;
	std::vector<int> starts(shape.experts);
	std::vector<int> ends(shape.experts);
	std::vector<int> starts_2sm(shape.experts);
	std::vector<int> ends_2sm(shape.experts);
	for (int expert = 0; expert < shape.experts; ++expert) {
		starts[expert] = expert * blocks_per_expert;
		ends[expert] = (expert + 1) * blocks_per_expert;
		starts_2sm[expert] =
			starts[expert] * Traits1Sm::TileK / Traits2Sm::TileK;
		ends_2sm[expert] =
			ends[expert] * Traits1Sm::TileK / Traits2Sm::TileK;
	}

	CUDA_CHECK(cudaMalloc(
		&inputs.k_starts, shape.experts * sizeof(int)));
	CUDA_CHECK(cudaMalloc(
		&inputs.k_ends, shape.experts * sizeof(int)));
	CUDA_CHECK(cudaMalloc(
		&inputs.k_starts_2sm, shape.experts * sizeof(int)));
	CUDA_CHECK(cudaMalloc(
		&inputs.k_ends_2sm, shape.experts * sizeof(int)));
	CUDA_CHECK(cudaMemcpy(
		inputs.k_starts, starts.data(), shape.experts * sizeof(int),
		cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(
		inputs.k_ends, ends.data(), shape.experts * sizeof(int),
		cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(
		inputs.k_starts_2sm, starts_2sm.data(),
		shape.experts * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(
		inputs.k_ends_2sm, ends_2sm.data(),
		shape.experts * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaDeviceSynchronize());
	return inputs;
}

// Explicit per-expert K-block plan (in Traits1Sm::TileK==Traits2Sm::TileK
// units). Unlike make_inputs() above (uniform tokens_per_expert), this lets
// a shape's experts hold zero blocks (empty expert), a highly uneven split
// (skew), or a prime/non-power-of-two count (exercises ceil-div K-split
// remainder handling — "tails" and repeated TMA_REDUCE_ADD accumulation).
// shape.tokens must equal sum(blocks_per_expert) * Traits1Sm::TileK.
static DeviceInputs make_inputs_blocks(
		const Shape& shape, const std::vector<int>& blocks_per_expert) {
	if (static_cast<int>(blocks_per_expert.size()) != shape.experts) {
		std::fprintf(
			stderr, "%s: blocks_per_expert.size() != shape.experts\n",
			shape.name);
		std::exit(1);
	}
	int total_blocks = 0;
	for (int blocks : blocks_per_expert) total_blocks += blocks;
	if (total_blocks * Traits1Sm::TileK != shape.tokens) {
		std::fprintf(
			stderr,
			"%s: blocks_per_expert sums to %d blocks, expected %d\n",
			shape.name, total_blocks, shape.tokens / Traits1Sm::TileK);
		std::exit(1);
	}

	DeviceInputs inputs;
	size_t dy_count = static_cast<size_t>(shape.tokens) * shape.hidden;
	size_t z_count =
		static_cast<size_t>(shape.tokens) * shape.intermediate;
	CUDA_CHECK(cudaMalloc(&inputs.dy, dy_count * sizeof(Element)));
	CUDA_CHECK(cudaMalloc(&inputs.z, z_count * sizeof(Element)));

	fill_bf16_kernel<<<1024, 256>>>(inputs.dy, dy_count, 0x2468acefu);
	fill_bf16_kernel<<<1024, 256>>>(inputs.z, z_count, 0x13579bdfu);
	CUDA_CHECK(cudaGetLastError());

	std::vector<int> starts(shape.experts);
	std::vector<int> ends(shape.experts);
	std::vector<int> starts_2sm(shape.experts);
	std::vector<int> ends_2sm(shape.experts);
	int running = 0;
	for (int expert = 0; expert < shape.experts; ++expert) {
		starts[expert] = running;
		running += blocks_per_expert[expert];
		ends[expert] = running;
		starts_2sm[expert] =
			starts[expert] * Traits1Sm::TileK / Traits2Sm::TileK;
		ends_2sm[expert] =
			ends[expert] * Traits1Sm::TileK / Traits2Sm::TileK;
	}

	CUDA_CHECK(cudaMalloc(
		&inputs.k_starts, shape.experts * sizeof(int)));
	CUDA_CHECK(cudaMalloc(
		&inputs.k_ends, shape.experts * sizeof(int)));
	CUDA_CHECK(cudaMalloc(
		&inputs.k_starts_2sm, shape.experts * sizeof(int)));
	CUDA_CHECK(cudaMalloc(
		&inputs.k_ends_2sm, shape.experts * sizeof(int)));
	CUDA_CHECK(cudaMemcpy(
		inputs.k_starts, starts.data(), shape.experts * sizeof(int),
		cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(
		inputs.k_ends, ends.data(), shape.experts * sizeof(int),
		cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(
		inputs.k_starts_2sm, starts_2sm.data(),
		shape.experts * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(
		inputs.k_ends_2sm, ends_2sm.data(),
		shape.experts * sizeof(int), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaDeviceSynchronize());
	return inputs;
}

template <typename Traits>
static auto make_one_sm_dyt_tma(
		const DeviceInputs& inputs, const Shape& shape) {
	auto tensor = make_tensor(
		make_gmem_ptr(inputs.dy),
		make_shape(shape.hidden, shape.tokens),
		make_stride(Int<1>{}, shape.hidden));
	return make_tma_copy(
		SM90_TMA_LOAD{}, tensor, typename Traits::SmemLayoutDYT_1{});
}

template <typename Traits>
static auto make_one_sm_z_tma(
		const DeviceInputs& inputs, const Shape& shape) {
	auto tensor = make_tensor(
		make_gmem_ptr(inputs.z),
		make_shape(shape.intermediate, shape.tokens),
		make_stride(Int<1>{}, shape.intermediate));
	return make_tma_copy(
		SM90_TMA_LOAD{}, tensor, typename Traits::SmemLayoutZ_1{});
}

template <typename Traits>
static auto make_two_sm_dyt_tma(
		const DeviceInputs& inputs, const Shape& shape) {
	auto tensor = make_tensor(
		make_gmem_ptr(inputs.dy),
		make_shape(shape.hidden, shape.tokens),
		make_stride(Int<1>{}, shape.hidden));
	return make_tma_copy_A_sm100(
		SM100_TMA_2SM_LOAD{}, tensor,
		typename Traits::SmemLayoutDYT_1{},
		typename Traits::TileShape{},
		typename Traits::TiledMma2Sm{});
}

template <typename Traits>
static auto make_two_sm_z_tma(
		const DeviceInputs& inputs, const Shape& shape) {
	auto tensor = make_tensor(
		make_gmem_ptr(inputs.z),
		make_shape(shape.intermediate, shape.tokens),
		make_stride(Int<1>{}, shape.intermediate));
	return make_tma_copy_B_sm100(
		SM100_TMA_2SM_LOAD{}, tensor,
		typename Traits::SmemLayoutZ_1{},
		typename Traits::TileShape{},
		typename Traits::TiledMma2Sm{});
}

template <typename Traits>
static auto make_da_tma(Element* output, const Shape& shape) {
	int rows = shape.experts * shape.hidden;
	auto tensor = make_tensor(
		make_gmem_ptr(output),
		make_shape(rows, shape.intermediate),
		make_stride(shape.intermediate, Int<1>{}));
	return make_tma_copy(
		SM90_TMA_REDUCE_ADD{},
		tensor,
		typename Traits::SmemLayoutStore{});
}

template <typename Kernel, typename... Args>
static void launch_two_sm(
		Kernel kernel,
		dim3 grid,
		dim3 block,
		size_t dynamic_smem,
		cudaStream_t stream,
		Args... args) {
	cudaLaunchConfig_t config{};
	config.gridDim = grid;
	config.blockDim = block;
	config.dynamicSmemBytes = dynamic_smem;
	config.stream = stream;

	cudaLaunchAttribute attribute{};
	attribute.id = cudaLaunchAttributeClusterDimension;
	attribute.val.clusterDim.x = 2;
	attribute.val.clusterDim.y = 1;
	attribute.val.clusterDim.z = 1;
	config.attrs = &attribute;
	config.numAttrs = 1;
	CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, args...));
}

template <typename TmaLoadDYT, typename TmaLoadZ, typename TmaReduceDA>
static void set_one_sm_attributes(
		TmaLoadDYT const&, TmaLoadZ const&, TmaReduceDA const&) {
	auto kernel = mlp3_one_sm_kernel<
		Traits1Sm, TmaLoadDYT, TmaLoadZ, TmaReduceDA>;
	CUDA_CHECK(cudaFuncSetAttribute(
		kernel,
		cudaFuncAttributeMaxDynamicSharedMemorySize,
		sizeof(Mlp3OneSmKernelSmem<Traits1Sm>)));
}

template <typename TmaLoadDYT, typename TmaLoadZ, typename TmaReduceDA>
static void set_two_sm_attributes(
		TmaLoadDYT const&, TmaLoadZ const&, TmaReduceDA const&) {
	auto kernel = mlp3_two_sm_kernel<
		Traits2Sm, TmaLoadDYT, TmaLoadZ, TmaReduceDA>;
	CUDA_CHECK(cudaFuncSetAttribute(
		kernel,
		cudaFuncAttributeMaxDynamicSharedMemorySize,
		sizeof(liger::Mlp3Smem2Sm<Traits2Sm>)));
	CUDA_CHECK(cudaFuncSetAttribute(
		kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
}

template <typename TmaLoadDYT, typename TmaLoadZ, typename TmaReduceDA>
static void launch_one_sm_case(
		TmaLoadDYT const& tma_dyt,
		TmaLoadZ const& tma_z,
		TmaReduceDA const& tma_da,
		const DeviceInputs& inputs,
		const Shape& shape,
		int outer_split,
		int grid,
		cudaStream_t stream,
		int k_split = 1) {
	int m_tiles = shape.hidden / Traits1Sm::TileM;
	int n_tiles = shape.intermediate / Traits1Sm::TileN;
	int rows = shape.experts * shape.hidden;
	auto kernel = mlp3_one_sm_kernel<
		Traits1Sm, TmaLoadDYT, TmaLoadZ, TmaReduceDA>;
	kernel<<<
		grid, Traits1Sm::NumThreads,
		sizeof(Mlp3OneSmKernelSmem<Traits1Sm>), stream>>>(
		tma_dyt, tma_z, tma_da,
		inputs.k_starts, inputs.k_ends,
		shape.experts, shape.hidden, shape.intermediate, shape.tokens,
		rows, m_tiles, n_tiles, outer_split, k_split);
	CUDA_CHECK(cudaGetLastError());
}

template <typename TmaLoadDYT, typename TmaLoadZ, typename TmaReduceDA>
static void launch_two_sm_case(
		TmaLoadDYT const& tma_dyt,
		TmaLoadZ const& tma_z,
		TmaReduceDA const& tma_da,
		const DeviceInputs& inputs,
		const Shape& shape,
		int outer_split,
		int grid,
		cudaStream_t stream,
		int k_split = 1) {
	int m_tiles = shape.hidden / Traits2Sm::TileM;
	int n_tiles = shape.intermediate / Traits2Sm::TileN;
	auto kernel = mlp3_two_sm_kernel<
		Traits2Sm, TmaLoadDYT, TmaLoadZ, TmaReduceDA>;
	launch_two_sm(
		kernel, dim3(grid), dim3(Traits2Sm::NumThreads),
		sizeof(liger::Mlp3Smem2Sm<Traits2Sm>), stream,
		tma_dyt, tma_z, tma_da,
		inputs.k_starts_2sm, inputs.k_ends_2sm,
		shape.experts, shape.hidden, shape.intermediate, shape.tokens,
		m_tiles, n_tiles, outer_split, k_split);
}

static std::vector<float> download(Element* device, size_t count) {
	std::vector<Element> packed(count);
	CUDA_CHECK(cudaMemcpy(
		packed.data(), device, count * sizeof(Element),
		cudaMemcpyDeviceToHost));
	std::vector<float> output(count);
	for (size_t i = 0; i < count; ++i)
		output[i] = static_cast<float>(packed[i]);
	return output;
}

static std::vector<float> cpu_reference(
		const DeviceInputs& inputs, const Shape& shape) {
	size_t dy_count = static_cast<size_t>(shape.tokens) * shape.hidden;
	size_t z_count =
		static_cast<size_t>(shape.tokens) * shape.intermediate;
	auto dy = download(inputs.dy, dy_count);
	auto z = download(inputs.z, z_count);

	std::vector<float> reference(
		static_cast<size_t>(shape.experts) * shape.hidden *
			shape.intermediate,
		0.0f);
	int tokens_per_expert = shape.tokens / shape.experts;
	for (int expert = 0; expert < shape.experts; ++expert) {
		int begin = expert * tokens_per_expert;
		int end = begin + tokens_per_expert;
		float* expert_output =
			reference.data() +
			static_cast<size_t>(expert) * shape.hidden *
				shape.intermediate;
		for (int token = begin; token < end; ++token) {
			const float* dy_row =
				dy.data() + static_cast<size_t>(token) * shape.hidden;
			const float* z_row =
				z.data() +
				static_cast<size_t>(token) * shape.intermediate;
			for (int hidden = 0; hidden < shape.hidden; ++hidden) {
				float value = dy_row[hidden];
				float* output_row =
					expert_output +
					static_cast<size_t>(hidden) * shape.intermediate;
				for (int intermediate = 0;
				     intermediate < shape.intermediate;
				     ++intermediate)
					output_row[intermediate] += value * z_row[intermediate];
			}
		}
	}
	return reference;
}

// Same accumulation as cpu_reference(), but with explicit per-expert
// [begin,end) token ranges derived from an arbitrary blocks_per_expert plan
// (see make_inputs_blocks) instead of an assumed uniform split. Experts with
// zero blocks correctly contribute an all-zero output slab.
static std::vector<float> cpu_reference_blocks(
		const DeviceInputs& inputs, const Shape& shape,
		const std::vector<int>& blocks_per_expert) {
	size_t dy_count = static_cast<size_t>(shape.tokens) * shape.hidden;
	size_t z_count =
		static_cast<size_t>(shape.tokens) * shape.intermediate;
	auto dy = download(inputs.dy, dy_count);
	auto z = download(inputs.z, z_count);

	std::vector<float> reference(
		static_cast<size_t>(shape.experts) * shape.hidden *
			shape.intermediate,
		0.0f);
	int running = 0;
	for (int expert = 0; expert < shape.experts; ++expert) {
		int begin = running * Traits1Sm::TileK;
		running += blocks_per_expert[expert];
		int end = running * Traits1Sm::TileK;
		float* expert_output =
			reference.data() +
			static_cast<size_t>(expert) * shape.hidden *
				shape.intermediate;
		for (int token = begin; token < end; ++token) {
			const float* dy_row =
				dy.data() + static_cast<size_t>(token) * shape.hidden;
			const float* z_row =
				z.data() +
				static_cast<size_t>(token) * shape.intermediate;
			for (int hidden = 0; hidden < shape.hidden; ++hidden) {
				float value = dy_row[hidden];
				float* output_row =
					expert_output +
					static_cast<size_t>(hidden) * shape.intermediate;
				for (int intermediate = 0;
				     intermediate < shape.intermediate;
				     ++intermediate)
					output_row[intermediate] += value * z_row[intermediate];
			}
		}
	}
	return reference;
}

struct ErrorStats {
	double mean_relative = 0.0;
	double max_relative = 0.0;
	double max_absolute = 0.0;
};

static ErrorStats compare(
		const std::vector<float>& actual,
		const std::vector<float>& expected) {
	ErrorStats stats;
	for (size_t i = 0; i < actual.size(); ++i) {
		double absolute =
			std::abs(static_cast<double>(actual[i]) - expected[i]);
		double relative =
			absolute /
			std::max(std::abs(static_cast<double>(expected[i])), 1e-3);
		stats.mean_relative += relative;
		stats.max_relative = std::max(stats.max_relative, relative);
		stats.max_absolute = std::max(stats.max_absolute, absolute);
	}
	stats.mean_relative /= actual.size();
	return stats;
}

static void check_stats(
		const char* label, const Shape& shape, const ErrorStats& stats) {
	std::printf(
		"CORRECTNESS,%s,%s,mean_rel=%.6f%%,max_rel=%.6f%%,max_abs=%.6g\n",
		label, shape.name,
		stats.mean_relative * 100.0,
		stats.max_relative * 100.0,
		stats.max_absolute);
	if (stats.mean_relative >= 0.01 || stats.max_relative >= 0.05) {
		std::fprintf(stderr, "%s failed correctness tolerance\n", label);
		std::exit(2);
	}
}

// Per-element relative error (as used by check_stats/compare above) is the
// right tool when two runs are expected to sum terms in the same order
// (e.g. outer_split only redistributes *which* CTA visits a tile, so its
// sum order is untouched). k_split, by design, evaluates an expert's K-loop
// as N independently-accumulated TMA_REDUCE_ADD partial sums instead of one
// pass, so its result differs from the k_split=1 baseline by ordinary
// floating-point reassociation noise. A per-element relative check with a
// small denominator floor is ill-conditioned wherever the reference value
// is near zero from cancellation (a near-zero dA cell blows up to a huge
// "relative error" from a very ordinary ~1e-2 absolute bf16 rounding
// difference). MoEBenchmark's own production check_bwd_correctness()
// (src/main_bwd.cpp) hits the same issue for its multi-pass split-K
// reductions and solves it by normalizing against the reference tensor's
// global max-magnitude instead of each element's own value; mirror that
// methodology here for the k_split comparisons.
static ErrorStats compare_absmax(
		const std::vector<float>& actual,
		const std::vector<float>& expected) {
	ErrorStats stats;
	double absmax = 0.0;
	for (float value : expected)
		absmax = std::max(absmax, std::abs(static_cast<double>(value)));
	double denom = std::max(absmax, 1e-6);
	for (size_t i = 0; i < actual.size(); ++i) {
		double absolute =
			std::abs(static_cast<double>(actual[i]) - expected[i]);
		double relative = absolute / denom;
		stats.mean_relative += relative;
		stats.max_relative = std::max(stats.max_relative, relative);
		stats.max_absolute = std::max(stats.max_absolute, absolute);
	}
	stats.mean_relative /= actual.size();
	return stats;
}

// Gates on mean relative error only (same as production's kMeanRelTol path):
// a handful of near-zero cancellation cells can legitimately dominate a
// max-relative metric without indicating any real accumulation bug.
static void check_stats_absmax(
		const char* label, const Shape& shape, const ErrorStats& stats) {
	std::printf(
		"CORRECTNESS,%s,%s,mean_rel_absmax=%.6f%%,max_rel_absmax=%.6f%%,"
		"max_abs=%.6g\n",
		label, shape.name,
		stats.mean_relative * 100.0,
		stats.max_relative * 100.0,
		stats.max_absolute);
	if (stats.mean_relative >= 0.05) {
		std::fprintf(stderr, "%s failed correctness tolerance\n", label);
		std::exit(2);
	}
}

static std::vector<float> run_one_sm_correctness(
		const Shape& shape,
		const DeviceInputs& inputs,
		int outer_split,
		int k_split = 1) {
	size_t output_count =
		static_cast<size_t>(shape.experts) * shape.hidden *
		shape.intermediate;
	Element* output = nullptr;
	CUDA_CHECK(cudaMalloc(&output, output_count * sizeof(Element)));
	CUDA_CHECK(cudaMemset(output, 0, output_count * sizeof(Element)));

	auto tma_dyt = make_one_sm_dyt_tma<Traits1Sm>(inputs, shape);
	auto tma_z = make_one_sm_z_tma<Traits1Sm>(inputs, shape);
	auto tma_da = make_da_tma<Traits1Sm>(output, shape);
	set_one_sm_attributes(tma_dyt, tma_z, tma_da);
	int total_cells =
		shape.experts * (shape.hidden / Traits1Sm::TileM) *
		outer_split;
	int grid = std::max(1, std::min(sm_count(), total_cells));
	launch_one_sm_case(
		tma_dyt, tma_z, tma_da, inputs, shape, outer_split, grid, 0,
		k_split);
	CUDA_CHECK(cudaDeviceSynchronize());
	auto result = download(output, output_count);
	CUDA_CHECK(cudaFree(output));
	return result;
}

static std::vector<float> run_two_sm_correctness(
		const Shape& shape,
		const DeviceInputs& inputs,
		int outer_split,
		int k_split = 1) {
	size_t output_count =
		static_cast<size_t>(shape.experts) * shape.hidden *
		shape.intermediate;
	Element* output = nullptr;
	CUDA_CHECK(cudaMalloc(&output, output_count * sizeof(Element)));
	CUDA_CHECK(cudaMemset(output, 0, output_count * sizeof(Element)));

	auto tma_dyt = make_two_sm_dyt_tma<Traits2Sm>(inputs, shape);
	auto tma_z = make_two_sm_z_tma<Traits2Sm>(inputs, shape);
	auto tma_da = make_da_tma<Traits2Sm>(output, shape);
	set_two_sm_attributes(tma_dyt, tma_z, tma_da);
	int total_cells =
		shape.experts * (shape.intermediate / Traits2Sm::TileN) *
		outer_split;
	int pairs = std::max(
		1, std::min(sm_count() / 2, total_cells));
	int grid = 2 * pairs;
	launch_two_sm_case(
		tma_dyt, tma_z, tma_da,
		inputs, shape, outer_split, grid, 0, k_split);
	CUDA_CHECK(cudaDeviceSynchronize());
	auto result = download(output, output_count);
	CUDA_CHECK(cudaFree(output));
	return result;
}

static void run_correctness() {
	const Shape small{"small", 512, 256, 256, 4};
	{
		auto inputs = make_inputs(small);
		auto reference = cpu_reference(inputs, small);
		auto one_sm = run_one_sm_correctness(small, inputs, 1);
		auto two_sm = run_two_sm_correctness(small, inputs, 1);
		check_stats("1sm", small, compare(one_sm, reference));
		check_stats("2sm", small, compare(two_sm, reference));
	}

	const Shape multi{"multi", 1024, 512, 512, 4};
	{
		auto inputs = make_inputs(multi);
		auto one_sm_split1 =
			run_one_sm_correctness(multi, inputs, 1);
		auto one_sm_split2 =
			run_one_sm_correctness(multi, inputs, 2);
		auto two_sm_split1 =
			run_two_sm_correctness(multi, inputs, 1);
		auto two_sm_split2 =
			run_two_sm_correctness(multi, inputs, 2);
		check_stats(
			"1sm-split2", multi,
			compare(one_sm_split2, one_sm_split1));
		check_stats(
			"2sm-split1", multi,
			compare(two_sm_split1, one_sm_split1));
		check_stats(
			"2sm-split2", multi,
			compare(two_sm_split2, one_sm_split1));
	}
}

// Targeted correctness for expert-distribution edge cases that
// run_correctness() above does not cover: empty experts (zero K-blocks),
// heavily skewed block counts, non-power-of-two ("tail") block counts that
// force a ceil-div remainder in the K-split path, and repeated
// TMA_REDUCE_ADD accumulation (k_split > 1) into the same dA output cell.
// hidden/intermediate are held at the same 256/256 single-tile shape as
// run_correctness()'s "small" case so only the expert/K-block plan varies.
static void run_edge_case_correctness() {
	struct EdgeCase {
		const char* name;
		int experts;
		std::vector<int> blocks_per_expert;  // TileK(=64)-unit K-blocks
	};

	const std::vector<EdgeCase> cases = {
		// One empty expert (index 0), one single-block "tail" expert
		// (index 1), one heavily skewed/dominant expert (index 2), one
		// mid-size expert (index 3). 0+1+5+2 = 8 blocks = 512 tokens.
		{"empty-skew", 4, {0, 1, 5, 2}},
		// All tokens routed to a single expert; the other three are
		// entirely empty (the extreme end of MoE load imbalance).
		{"all-but-one-empty", 4, {0, 0, 0, 8}},
		// Eight experts, four of them empty, remaining four holding a
		// prime/non-power-of-two block count each (tail-heavy skew with
		// more grid-stride cells than active work items).
		{"many-empty-prime", 8, {0, 1, 0, 3, 0, 2, 0, 1}},
	};

	for (const auto& edge : cases) {
		int total_blocks = 0;
		for (int blocks : edge.blocks_per_expert) total_blocks += blocks;
		// hidden=512/intermediate=512 matches run_correctness()'s "multi"
		// shape tile geometry (2 M-tiles and 2 N-tiles on both the 1SM
		// N-split and 2SM M-split axes), so outer_split=2 below exercises
		// only the expert/K-block distribution being varied here, not an
		// unrelated tile-count-vs-split-factor mismatch on either kernel.
		Shape shape{
			edge.name, total_blocks * Traits1Sm::TileK, 512, 512,
			edge.experts};

		auto inputs = make_inputs_blocks(shape, edge.blocks_per_expert);
		auto reference =
			cpu_reference_blocks(inputs, shape, edge.blocks_per_expert);

		auto one_sm_k1 =
			run_one_sm_correctness(shape, inputs, /*outer_split=*/1,
				/*k_split=*/1);
		auto two_sm_k1 =
			run_two_sm_correctness(shape, inputs, /*outer_split=*/1,
				/*k_split=*/1);
		check_stats("1sm", shape, compare(one_sm_k1, reference));
		check_stats("2sm", shape, compare(two_sm_k1, reference));

		// outer_split (M-tile grid-parallelism) with skewed/empty experts.
		auto one_sm_split2 =
			run_one_sm_correctness(shape, inputs, /*outer_split=*/2,
				/*k_split=*/1);
		auto two_sm_split2 =
			run_two_sm_correctness(shape, inputs, /*outer_split=*/2,
				/*k_split=*/1);
		check_stats(
			"1sm-split2", shape, compare(one_sm_split2, one_sm_k1));
		check_stats(
			"2sm-split2", shape, compare(two_sm_split2, one_sm_k1));

		// k_split (repeated TMA_REDUCE_ADD accumulation into the same dA
		// cell). k_split=3 does not evenly divide the 5- or 3-block
		// experts above, exercising the ceil-div remainder tail. Splitting
		// K changes summation order vs. the k_split=1 baseline, so this
		// uses the absmax-normalized comparison (see compare_absmax).
		for (int k_split : {2, 3}) {
			auto one_sm_ksplit = run_one_sm_correctness(
				shape, inputs, /*outer_split=*/1, k_split);
			auto two_sm_ksplit = run_two_sm_correctness(
				shape, inputs, /*outer_split=*/1, k_split);
			char label_1sm[32];
			char label_2sm[32];
			std::snprintf(
				label_1sm, sizeof(label_1sm), "1sm-ksplit%d", k_split);
			std::snprintf(
				label_2sm, sizeof(label_2sm), "2sm-ksplit%d", k_split);
			check_stats_absmax(
				label_1sm, shape,
				compare_absmax(one_sm_ksplit, one_sm_k1));
			check_stats_absmax(
				label_2sm, shape,
				compare_absmax(two_sm_ksplit, one_sm_k1));
		}
	}
}


template <typename Launch>
static double time_launches(
		Launch launch,
		Element* output,
		size_t output_bytes,
		uint32_t* eviction,
		size_t eviction_count,
		cudaStream_t stream) {
	int warmup = env_int("MLP3_WARMUP", 5);
	int iterations = env_int("MLP3_ITERS", 30);

	for (int iteration = 0; iteration < warmup; ++iteration) {
		CUDA_CHECK(cudaMemsetAsync(output, 0, output_bytes, stream));
		evict_l2_kernel<<<1024, 256, 0, stream>>>(
			eviction, eviction_count, iteration);
		launch();
	}
	CUDA_CHECK(cudaStreamSynchronize(stream));

	cudaEvent_t start;
	cudaEvent_t stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));
	std::vector<float> samples;
	samples.reserve(iterations);

	for (int iteration = 0; iteration < iterations; ++iteration) {
		CUDA_CHECK(cudaMemsetAsync(output, 0, output_bytes, stream));
		evict_l2_kernel<<<1024, 256, 0, stream>>>(
			eviction, eviction_count, 0x10000u + iteration);
		CUDA_CHECK(cudaEventRecord(start, stream));
		launch();
		CUDA_CHECK(cudaEventRecord(stop, stream));
		CUDA_CHECK(cudaEventSynchronize(stop));
		float milliseconds = 0.0f;
		CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
		samples.push_back(milliseconds);
	}

	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));
	return median(std::move(samples));
}

static BenchmarkResult benchmark_one_sm(
		const Shape& shape,
		const DeviceInputs& inputs,
		Element* output,
		uint32_t* eviction,
		size_t eviction_count,
		cudaStream_t stream) {
	auto tma_dyt = make_one_sm_dyt_tma<Traits1Sm>(inputs, shape);
	auto tma_z = make_one_sm_z_tma<Traits1Sm>(inputs, shape);
	auto tma_da = make_da_tma<Traits1Sm>(output, shape);
	set_one_sm_attributes(tma_dyt, tma_z, tma_da);
	size_t output_bytes =
		static_cast<size_t>(shape.experts) * shape.hidden *
		shape.intermediate * sizeof(Element);
	int m_tiles = shape.hidden / Traits1Sm::TileM;
	int n_tiles = shape.intermediate / Traits1Sm::TileN;

	BenchmarkResult best;
	for (int split : split_candidates(
		     n_tiles, "MLP3_FIXED_1SM_SPLIT")) {
		int cells = shape.experts * m_tiles * split;
		int grid = std::max(1, std::min(sm_count(), cells));
		auto launch = [&]() {
			launch_one_sm_case(
				tma_dyt, tma_z, tma_da, inputs, shape,
				split, grid, stream);
		};
		double ms = time_launches(
			launch, output, output_bytes,
			eviction, eviction_count, stream);
		double throughput = tflops(shape, ms);
		std::printf(
			"SWEEP,1sm,%s,split=%d,grid=%d,ms=%.6f,tflops=%.3f\n",
			shape.name, split, grid, ms, throughput);
		if (throughput > best.tflops)
			best = {ms, throughput, split, grid};
	}
	return best;
}

static BenchmarkResult benchmark_two_sm(
		const Shape& shape,
		const DeviceInputs& inputs,
		Element* output,
		uint32_t* eviction,
		size_t eviction_count,
		cudaStream_t stream) {
	auto tma_dyt = make_two_sm_dyt_tma<Traits2Sm>(inputs, shape);
	auto tma_z = make_two_sm_z_tma<Traits2Sm>(inputs, shape);
	auto tma_da = make_da_tma<Traits2Sm>(output, shape);
	set_two_sm_attributes(tma_dyt, tma_z, tma_da);
	size_t output_bytes =
		static_cast<size_t>(shape.experts) * shape.hidden *
		shape.intermediate * sizeof(Element);
	int m_tiles = shape.hidden / Traits2Sm::TileM;
	int n_tiles = shape.intermediate / Traits2Sm::TileN;

	BenchmarkResult best;
	for (int split : split_candidates(
		     m_tiles, "MLP3_FIXED_2SM_SPLIT")) {
		int cells = shape.experts * n_tiles * split;
		int pairs = std::max(1, std::min(sm_count() / 2, cells));
		int grid = 2 * pairs;
		auto launch = [&]() {
			launch_two_sm_case(
				tma_dyt, tma_z, tma_da, inputs, shape,
				split, grid, stream);
		};
		double ms = time_launches(
			launch, output, output_bytes,
			eviction, eviction_count, stream);
		double throughput = tflops(shape, ms);
		std::printf(
			"SWEEP,2sm,%s,split=%d,grid=%d,ms=%.6f,tflops=%.3f\n",
			shape.name, split, grid, ms, throughput);
		if (throughput > best.tflops)
			best = {ms, throughput, split, grid};
	}
	return best;
}

static void run_benchmarks() {
	const std::vector<Shape> shapes = {
		{"Qwen3-30B-A3B", 8192, 2048, 768, 8},
		{"Qwen3-235B-A22B", 8192, 4096, 1536, 8},
		{"Qwen3.5-122B-A10B", 8192, 3072, 1024, 8},
		{"Llama-4-Scout-17B-16E", 8192, 5120, 8192, 8},
		{"Mixtral-8x7B", 8192, 4096, 14336, 8},
		{"Mixtral-8x22B", 8192, 6144, 16384, 8},
	};

	uint32_t* eviction = nullptr;
	CUDA_CHECK(cudaMalloc(&eviction, kL2EvictionBytes));
	size_t eviction_count = kL2EvictionBytes / sizeof(uint32_t);
	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	double log_speedup = 0.0;
	int measured_shapes = 0;
	const char* shape_filter = std::getenv("MLP3_SHAPE");
	for (const auto& shape : shapes) {
		if (shape_filter != nullptr &&
		    std::string(shape.name) != shape_filter)
			continue;
		if (shape.hidden % 256 != 0 ||
		    shape.intermediate % Traits2Sm::TileN != 0) {
			std::fprintf(
				stderr, "%s is not exactly tiled by the pilot\n", shape.name);
			std::exit(3);
		}

		auto inputs = make_inputs(shape);
		size_t output_bytes =
			static_cast<size_t>(shape.experts) * shape.hidden *
			shape.intermediate * sizeof(Element);
		Element* output = nullptr;
		CUDA_CHECK(cudaMalloc(&output, output_bytes));

		auto one_sm = benchmark_one_sm(
			shape, inputs, output, eviction, eviction_count, stream);
		auto two_sm = benchmark_two_sm(
			shape, inputs, output, eviction, eviction_count, stream);
		double speedup = one_sm.ms / two_sm.ms;
		log_speedup += std::log(speedup);
		++measured_shapes;
		std::printf(
			"RESULT,%s,1sm_ms=%.6f,1sm_tflops=%.3f,1sm_split=%d,"
			"2sm_ms=%.6f,2sm_tflops=%.3f,2sm_split=%d,speedup=%.6f\n",
			shape.name,
			one_sm.ms, one_sm.tflops, one_sm.split,
			two_sm.ms, two_sm.tflops, two_sm.split,
			speedup);
		CUDA_CHECK(cudaFree(output));
	}

	std::printf(
		"GEOMEAN,speedup=%.6f\n",
		std::exp(log_speedup / measured_shapes));
	CUDA_CHECK(cudaStreamDestroy(stream));
	CUDA_CHECK(cudaFree(eviction));
}

} // namespace bench_mlp3_2sm

int main(int argc, char** argv) {
	int device = 0;
	cudaDeviceProp properties{};
	CUDA_CHECK(cudaGetDevice(&device));
	CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
	if (properties.major != 10 || properties.minor != 0) {
		std::fprintf(
			stderr,
			"SM100 required; found compute capability %d.%d\n",
			properties.major,
			properties.minor);
		return 1;
	}

	std::printf(
		"GPU,%s,sm_count=%d,l2_bytes=%d,2sm_tile=256x%dx%d,"
		"stages=%d,epi_n=%d,acc_stages=%d,compact_epilogue=%d,"
		"early_release=%d,2sm_smem_bytes=%zu\n",
		properties.name,
		properties.multiProcessorCount,
		properties.l2CacheSize,
		bench_mlp3_2sm::Traits2Sm::TileN,
		bench_mlp3_2sm::Traits2Sm::TileK,
		bench_mlp3_2sm::Traits2Sm::Stages,
		bench_mlp3_2sm::Traits2Sm::EpiChunkN,
		bench_mlp3_2sm::Traits2Sm::AccStages,
		bench_mlp3_2sm::Traits2Sm::CompactEpilogue ? 1 : 0,
		bench_mlp3_2sm::Traits2Sm::EarlyTmemRelease ? 1 : 0,
		sizeof(liger::Mlp3Smem2Sm<bench_mlp3_2sm::Traits2Sm>));
	if (std::getenv("MLP3_SKIP_CORRECTNESS") == nullptr) {
		bench_mlp3_2sm::run_correctness();
		bench_mlp3_2sm::run_edge_case_correctness();
	}
	if (argc == 1 || std::string(argv[1]) != "--correctness-only")
		bench_mlp3_2sm::run_benchmarks();
	return 0;
}
