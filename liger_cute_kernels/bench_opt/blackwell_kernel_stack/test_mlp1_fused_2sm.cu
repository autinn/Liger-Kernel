// Correctness and throughput experiment for the SM100 paired-CTA MLP1 path.
//
// The joined UMMA is 128x256. CUTLASS partitions its C layout as
// [peer=2, M/2=64, N=256], so "64x256 per SM" is a peer-local description,
// not an unsupported 2SM instruction with M=64.

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <cute/tensor.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cute/atom/copy_traits_sm100_tma.hpp>
#include <cutlass/numeric_types.h>

#include "mlp1_fused_2sm.cuh"

using namespace cute;
using Element = cutlass::bfloat16_t;

#ifndef LIGER_MLP1_2SM_STAGES
#define LIGER_MLP1_2SM_STAGES 4
#endif

#ifndef LIGER_MLP1_2SM_EPI_CHUNK_N
#define LIGER_MLP1_2SM_EPI_CHUNK_N 64
#endif

using Traits = liger::Mlp1Traits2Sm<
	Element, /*TileM=*/128, /*TileN=*/256, /*TileK=*/64,
	/*Stages=*/LIGER_MLP1_2SM_STAGES,
	/*EpiChunkN=*/LIGER_MLP1_2SM_EPI_CHUNK_N,
	/*AccStages=*/2>;
using Smem = liger::Mlp1Fused2SmSmem<Traits>;

#define CUDA_OK(expr)                                                         \
	do {                                                                      \
		cudaError_t _e = (expr);                                              \
		ASSERT_EQ(_e, cudaSuccess) << #expr << ": " << cudaGetErrorString(_e); \
	} while (0)

template <
	typename TmaLoadX,
	typename TmaLoadW,
	typename TmaStoreZ>
__global__ void __launch_bounds__(Traits::NumThreads, 1)
mlp1_fused_2sm_test_kernel(
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
	extern __shared__ char raw_smem[];
	auto& smem = *reinterpret_cast<Smem*>(raw_smem);
	if (gridDim.x % Traits::ClusterM != 0)
		__trap();
	liger::mlp1_fused_2sm<Traits>(
		smem, tma_load_x, tma_load_b, tma_load_c, tma_store_z,
		expert_ids, num_tokens, hidden_dim, total_n_rows,
		num_m_tiles, num_n_tiles);
}

struct Mlp1Shape {
	int num_tokens;
	int hidden_dim;
	int intermediate_dim;
	int num_experts;
};

struct DevBf16 {
	Element* ptr = nullptr;
	size_t n = 0;
	~DevBf16() {
		if (ptr)
			cudaFree(ptr);
	}
};

static inline float bf16_round(float x) {
	return float(Element(x));
}

static void upload_bf16(
		DevBf16& dst,
		const std::vector<float>& src) {
	std::vector<Element> rounded(src.size());
	for (size_t i = 0; i < src.size(); ++i)
		rounded[i] = Element(src[i]);
	dst.n = src.size();
	cudaMalloc(&dst.ptr, dst.n * sizeof(Element));
	cudaMemcpy(
		dst.ptr, rounded.data(), dst.n * sizeof(Element),
		cudaMemcpyHostToDevice);
}

struct Inputs {
	std::vector<float> X;
	std::vector<float> B;
	std::vector<float> C;
	std::vector<int> expert_ids;
	DevBf16 dX;
	DevBf16 dB;
	DevBf16 dC;
	int* d_expert_ids = nullptr;
	int num_m_tiles = 0;
	int num_n_tiles = 0;
	int total_n_rows = 0;

	~Inputs() {
		if (d_expert_ids)
			cudaFree(d_expert_ids);
	}
};

static void make_inputs(
		const Mlp1Shape& shape,
		Inputs& in,
		unsigned seed) {
	std::mt19937 rng(seed);
	std::normal_distribution<float> normal(0.0f, 1.0f);
	auto fill = [&](std::vector<float>& values, size_t count) {
		values.resize(count);
		for (float& value : values)
			value = bf16_round(normal(rng));
	};

	fill(in.X, static_cast<size_t>(shape.num_tokens) * shape.hidden_dim);
	fill(
		in.B,
		static_cast<size_t>(shape.num_experts) *
			shape.intermediate_dim * shape.hidden_dim);
	fill(
		in.C,
		static_cast<size_t>(shape.num_experts) *
			shape.intermediate_dim * shape.hidden_dim);

	in.num_m_tiles =
		(shape.num_tokens + Traits::TileM - 1) / Traits::TileM;
	in.num_n_tiles = shape.intermediate_dim / Traits::TileN;
	in.total_n_rows = shape.num_experts * shape.intermediate_dim;
	in.expert_ids.resize(in.num_m_tiles);
	for (int m = 0; m < in.num_m_tiles; ++m)
		in.expert_ids[m] = m % shape.num_experts;

	upload_bf16(in.dX, in.X);
	upload_bf16(in.dB, in.B);
	upload_bf16(in.dC, in.C);
	cudaMalloc(
		&in.d_expert_ids,
		in.expert_ids.size() * sizeof(int));
	cudaMemcpy(
		in.d_expert_ids, in.expert_ids.data(),
		in.expert_ids.size() * sizeof(int),
		cudaMemcpyHostToDevice);
}

static void make_bench_inputs(
		const Mlp1Shape& shape,
		Inputs& in) {
	in.num_m_tiles =
		(shape.num_tokens + Traits::TileM - 1) / Traits::TileM;
	in.num_n_tiles = shape.intermediate_dim / Traits::TileN;
	in.total_n_rows = shape.num_experts * shape.intermediate_dim;

	auto allocate = [](DevBf16& dst, size_t count) {
		dst.n = count;
		CUDA_OK(cudaMalloc(&dst.ptr, count * sizeof(Element)));
		CUDA_OK(cudaMemset(dst.ptr, 0x3c, count * sizeof(Element)));
	};
	allocate(
		in.dX,
		static_cast<size_t>(shape.num_tokens) * shape.hidden_dim);
	allocate(
		in.dB,
		static_cast<size_t>(in.total_n_rows) * shape.hidden_dim);
	allocate(
		in.dC,
		static_cast<size_t>(in.total_n_rows) * shape.hidden_dim);

	in.expert_ids.resize(in.num_m_tiles);
	for (int m = 0; m < in.num_m_tiles; ++m)
		in.expert_ids[m] = m % shape.num_experts;
	CUDA_OK(cudaMalloc(
		&in.d_expert_ids,
		in.expert_ids.size() * sizeof(int)));
	CUDA_OK(cudaMemcpy(
		in.d_expert_ids, in.expert_ids.data(),
		in.expert_ids.size() * sizeof(int),
		cudaMemcpyHostToDevice));
}

static std::vector<float> cpu_reference(
		const Inputs& in,
		const Mlp1Shape& shape) {
	int padded_tokens = in.num_m_tiles * Traits::TileM;
	std::vector<float> out(
		static_cast<size_t>(padded_tokens) * shape.intermediate_dim,
		0.0f);
	for (int m = 0; m < in.num_m_tiles; ++m) {
		int expert = in.expert_ids[m];
		int row_begin = m * Traits::TileM;
		int row_end = std::min(
			row_begin + Traits::TileM, shape.num_tokens);
		for (int row = row_begin; row < row_end; ++row) {
			for (int col = 0; col < shape.intermediate_dim; ++col) {
				float u = 0.0f;
				float v = 0.0f;
				const float* x =
					&in.X[static_cast<size_t>(row) * shape.hidden_dim];
				const float* b =
					&in.B[(static_cast<size_t>(expert) *
							shape.intermediate_dim +
						col) *
						shape.hidden_dim];
				const float* c =
					&in.C[(static_cast<size_t>(expert) *
							shape.intermediate_dim +
						col) *
						shape.hidden_dim];
				for (int k = 0; k < shape.hidden_dim; ++k) {
					u += x[k] * b[k];
					v += x[k] * c[k];
				}
				float silu = u / (1.0f + std::exp(-u));
				out[static_cast<size_t>(row) *
						shape.intermediate_dim +
					col] = silu * v;
			}
		}
	}
	return out;
}

struct ErrorStats {
	float max_abs = 0.0f;
	float mean_rel = 0.0f;
	float max_rel = 0.0f;
};

static ErrorStats compare(
		const std::vector<float>& got,
		const std::vector<float>& expected) {
	constexpr float atol = 1.0e-3f;
	ErrorStats stats;
	for (size_t i = 0; i < expected.size(); ++i) {
		float abs_error = std::fabs(got[i] - expected[i]);
		float rel_error =
			abs_error / std::max(std::fabs(expected[i]), atol);
		stats.max_abs = std::max(stats.max_abs, abs_error);
		stats.max_rel = std::max(stats.max_rel, rel_error);
		stats.mean_rel += rel_error;
	}
	stats.mean_rel /= expected.size();
	return stats;
}

static int sm_count() {
	int device = 0;
	cudaDeviceProp properties{};
	if (cudaGetDevice(&device) != cudaSuccess ||
		cudaGetDeviceProperties(&properties, device) != cudaSuccess)
		return 0;
	return properties.multiProcessorCount;
}

static int cluster_pair_count(int total_cells) {
	int max_pairs = std::max(
		1, std::min(sm_count() / Traits::ClusterM, total_cells));
	if (const char* value = std::getenv("MLP1_2SM_PAIRS")) {
		int requested = std::atoi(value);
		if (requested > 0)
			return std::min(requested, max_pairs);
	}
	return max_pairs;
}

template <typename Kernel, typename... Args>
static void launch_cluster(
		Kernel kernel,
		int grid_x,
		size_t smem_size,
		Args&&... args) {
	cudaLaunchConfig_t config{};
	config.gridDim = dim3(grid_x);
	config.blockDim = dim3(Traits::NumThreads);
	config.dynamicSmemBytes = smem_size;
	config.stream = nullptr;
	cudaLaunchAttribute cluster_attribute{};
	cluster_attribute.id = cudaLaunchAttributeClusterDimension;
	cluster_attribute.val.clusterDim.x = Traits::ClusterM;
	cluster_attribute.val.clusterDim.y = 1;
	cluster_attribute.val.clusterDim.z = 1;
	config.attrs = &cluster_attribute;
	config.numAttrs = 1;
	CUDA_OK(cudaLaunchKernelEx(
		&config, kernel, std::forward<Args>(args)...));
}

template <typename TmaLoadX, typename TmaLoadW, typename TmaStoreZ>
static auto get_kernel() {
	return mlp1_fused_2sm_test_kernel<
		TmaLoadX, TmaLoadW, TmaStoreZ>;
}

static bool blackwell_available() {
	int device = 0;
	cudaDeviceProp properties{};
	if (cudaGetDevice(&device) != cudaSuccess ||
		cudaGetDeviceProperties(&properties, device) != cudaSuccess)
		return false;
	return properties.major == 10;
}

static void run_correctness(const Mlp1Shape& shape) {
	ASSERT_EQ(shape.intermediate_dim % Traits::TileN, 0);
	ASSERT_EQ(shape.hidden_dim % Traits::TileK, 0);

	Inputs in;
	make_inputs(shape, in, 1234);
	int padded_tokens = in.num_m_tiles * Traits::TileM;
	size_t output_elements =
		static_cast<size_t>(padded_tokens) * shape.intermediate_dim;
	Element* dZ = nullptr;
	CUDA_OK(cudaMalloc(&dZ, output_elements * sizeof(Element)));
	CUDA_OK(cudaMemset(dZ, 0, output_elements * sizeof(Element)));

	auto tX = make_tensor(
		make_gmem_ptr(in.dX.ptr),
		make_shape(shape.num_tokens, shape.hidden_dim),
		make_stride(shape.hidden_dim, Int<1>{}));
	auto tB = make_tensor(
		make_gmem_ptr(in.dB.ptr),
		make_shape(in.total_n_rows, shape.hidden_dim),
		make_stride(shape.hidden_dim, Int<1>{}));
	auto tC = make_tensor(
		make_gmem_ptr(in.dC.ptr),
		make_shape(in.total_n_rows, shape.hidden_dim),
		make_stride(shape.hidden_dim, Int<1>{}));
	auto tZ = make_tensor(
		make_gmem_ptr(dZ),
		make_shape(padded_tokens, shape.intermediate_dim),
		make_stride(shape.intermediate_dim, Int<1>{}));

	auto tma_x = make_tma_copy_A_sm100(
		SM100_TMA_2SM_LOAD{}, tX,
		typename Traits::SmemLayoutX_1{},
		typename Traits::TileShape{},
		typename Traits::TiledMma2Sm{});
	auto tma_b = make_tma_copy_B_sm100(
		SM100_TMA_2SM_LOAD{}, tB,
		typename Traits::SmemLayoutW_1{},
		typename Traits::TileShape{},
		typename Traits::TiledMma2Sm{});
	auto tma_c = make_tma_copy_B_sm100(
		SM100_TMA_2SM_LOAD{}, tC,
		typename Traits::SmemLayoutW_1{},
		typename Traits::TileShape{},
		typename Traits::TiledMma2Sm{});
	auto tma_z = make_tma_copy(
		SM90_TMA_STORE{}, tZ,
		typename Traits::SmemLayoutStoreSlot{});

	auto kernel = get_kernel<
		decltype(tma_x), decltype(tma_b), decltype(tma_z)>();
	size_t smem_size = sizeof(Smem);
	CUDA_OK(cudaFuncSetAttribute(
		kernel,
		cudaFuncAttributeMaxDynamicSharedMemorySize,
		smem_size));
	CUDA_OK(cudaFuncSetAttribute(
		kernel,
		cudaFuncAttributeNonPortableClusterSizeAllowed,
		1));

	int total_cells = in.num_m_tiles * in.num_n_tiles;
	int pairs = cluster_pair_count(total_cells);
	launch_cluster(
		kernel, pairs * Traits::ClusterM, smem_size,
		tma_x, tma_b, tma_c, tma_z, in.d_expert_ids,
		shape.num_tokens, shape.hidden_dim, in.total_n_rows,
		in.num_m_tiles, in.num_n_tiles);
	CUDA_OK(cudaGetLastError());
	CUDA_OK(cudaDeviceSynchronize());

	std::vector<Element> device_output(output_elements);
	CUDA_OK(cudaMemcpy(
		device_output.data(), dZ,
		output_elements * sizeof(Element),
		cudaMemcpyDeviceToHost));
	cudaFree(dZ);
	std::vector<float> got(output_elements);
	for (size_t i = 0; i < output_elements; ++i)
		got[i] = float(device_output[i]);
	auto expected = cpu_reference(in, shape);
	auto error = compare(got, expected);

	printf(
		"[mlp1-2sm T=%d H=%d I=%d E=%d] "
		"mean_rel=%.3f%% max_rel=%.3f%% max_abs=%.3g\n",
		shape.num_tokens, shape.hidden_dim,
		shape.intermediate_dim, shape.num_experts,
		error.mean_rel * 100.0f, error.max_rel * 100.0f,
		error.max_abs);
	EXPECT_LT(error.mean_rel, 0.01f);
	EXPECT_LT(error.max_rel, 0.05f);
}

struct BenchConfig {
	int warmup = 10;
	int iterations = 50;
};

static double median_ms(std::vector<float>& samples) {
	std::sort(samples.begin(), samples.end());
	size_t count = samples.size();
	return count % 2 == 1
		? samples[count / 2]
		: 0.5 * (samples[count / 2 - 1] + samples[count / 2]);
}

static double tflops(const Mlp1Shape& shape, double milliseconds) {
	double operations =
		4.0 * shape.num_tokens * shape.hidden_dim *
		shape.intermediate_dim;
	return operations / (milliseconds * 1.0e-3) / 1.0e12;
}

static void run_benchmark(
		const Mlp1Shape& shape,
		const BenchConfig& config) {
	Inputs in;
	make_bench_inputs(shape, in);
	int padded_tokens = in.num_m_tiles * Traits::TileM;
	size_t output_elements =
		static_cast<size_t>(padded_tokens) * shape.intermediate_dim;
	Element* dZ = nullptr;
	CUDA_OK(cudaMalloc(&dZ, output_elements * sizeof(Element)));
	CUDA_OK(cudaMemset(dZ, 0, output_elements * sizeof(Element)));

	auto tX = make_tensor(
		make_gmem_ptr(in.dX.ptr),
		make_shape(shape.num_tokens, shape.hidden_dim),
		make_stride(shape.hidden_dim, Int<1>{}));
	auto tB = make_tensor(
		make_gmem_ptr(in.dB.ptr),
		make_shape(in.total_n_rows, shape.hidden_dim),
		make_stride(shape.hidden_dim, Int<1>{}));
	auto tC = make_tensor(
		make_gmem_ptr(in.dC.ptr),
		make_shape(in.total_n_rows, shape.hidden_dim),
		make_stride(shape.hidden_dim, Int<1>{}));
	auto tZ = make_tensor(
		make_gmem_ptr(dZ),
		make_shape(padded_tokens, shape.intermediate_dim),
		make_stride(shape.intermediate_dim, Int<1>{}));
	auto tma_x = make_tma_copy_A_sm100(
		SM100_TMA_2SM_LOAD{}, tX,
		typename Traits::SmemLayoutX_1{},
		typename Traits::TileShape{},
		typename Traits::TiledMma2Sm{});
	auto tma_b = make_tma_copy_B_sm100(
		SM100_TMA_2SM_LOAD{}, tB,
		typename Traits::SmemLayoutW_1{},
		typename Traits::TileShape{},
		typename Traits::TiledMma2Sm{});
	auto tma_c = make_tma_copy_B_sm100(
		SM100_TMA_2SM_LOAD{}, tC,
		typename Traits::SmemLayoutW_1{},
		typename Traits::TileShape{},
		typename Traits::TiledMma2Sm{});
	auto tma_z = make_tma_copy(
		SM90_TMA_STORE{}, tZ,
		typename Traits::SmemLayoutStoreSlot{});

	auto kernel = get_kernel<
		decltype(tma_x), decltype(tma_b), decltype(tma_z)>();
	size_t smem_size = sizeof(Smem);
	CUDA_OK(cudaFuncSetAttribute(
		kernel,
		cudaFuncAttributeMaxDynamicSharedMemorySize,
		smem_size));
	CUDA_OK(cudaFuncSetAttribute(
		kernel,
		cudaFuncAttributeNonPortableClusterSizeAllowed,
		1));
	int total_cells = in.num_m_tiles * in.num_n_tiles;
	int pairs = cluster_pair_count(total_cells);
	int grid_x = pairs * Traits::ClusterM;

	auto launch = [&]() {
		launch_cluster(
			kernel, grid_x, smem_size,
			tma_x, tma_b, tma_c, tma_z, in.d_expert_ids,
			shape.num_tokens, shape.hidden_dim, in.total_n_rows,
			in.num_m_tiles, in.num_n_tiles);
	};
	for (int i = 0; i < config.warmup; ++i)
		launch();
	CUDA_OK(cudaDeviceSynchronize());

	cudaEvent_t start;
	cudaEvent_t stop;
	CUDA_OK(cudaEventCreate(&start));
	CUDA_OK(cudaEventCreate(&stop));
	std::vector<float> samples;
	samples.reserve(config.iterations);
	for (int i = 0; i < config.iterations; ++i) {
		CUDA_OK(cudaEventRecord(start));
		launch();
		CUDA_OK(cudaEventRecord(stop));
		CUDA_OK(cudaEventSynchronize(stop));
		float milliseconds = 0.0f;
		CUDA_OK(cudaEventElapsedTime(
			&milliseconds, start, stop));
		samples.push_back(milliseconds);
	}
	CUDA_OK(cudaEventDestroy(start));
	CUDA_OK(cudaEventDestroy(stop));
	double milliseconds = median_ms(samples);
	printf(
		"[mlp1-2sm-bench s=%d epi=%d T=%-5d H=%d I=%d E=%d] "
		"peak %7.2f TFLOPS @ %7.4f ms "
		"(%d pairs / %d CTAs / %d SMs)\n",
		Traits::Stages, Traits::EpiChunkN,
		shape.num_tokens, shape.hidden_dim,
		shape.intermediate_dim, shape.num_experts,
		tflops(shape, milliseconds), milliseconds,
		pairs, grid_x, sm_count());
	cudaFree(dZ);
}

static const std::vector<Mlp1Shape> kCorrectnessShapes = {
	{128, 256, 256, 1},
	{128, 512, 512, 1},
	{256, 256, 512, 2},
	{384, 256, 256, 3},
};

static const std::vector<Mlp1Shape> kSquareBenchShapes = {
	{2048, 4096, 4096, 8},
	{4096, 4096, 4096, 8},
	{8192, 4096, 4096, 8},
	{16384, 4096, 4096, 8},
};

static const std::vector<Mlp1Shape> kModelBenchShapes = {
	{8192, 2048, 768, 8},
	{8192, 4096, 1536, 8},
	{8192, 2048, 512, 8},
	{8192, 3072, 1024, 8},
	{8192, 4096, 14336, 8},
	{8192, 6144, 16384, 8},
	{8192, 5120, 8192, 8},
};

TEST(Mlp1Fused2Sm, Correctness) {
	if (!blackwell_available())
		GTEST_SKIP() << "requires an sm_100 (Blackwell) GPU";
	for (const auto& shape : kCorrectnessShapes)
		run_correctness(shape);
}

TEST(Mlp1Fused2Sm, NoRegisterSpill) {
	if (!blackwell_available())
		GTEST_SKIP() << "requires an sm_100 (Blackwell) GPU";

	const Mlp1Shape shape{128, 512, 512, 1};
	Inputs in;
	make_inputs(shape, in, 1);
	Element* dZ = nullptr;
	CUDA_OK(cudaMalloc(
		&dZ,
		static_cast<size_t>(shape.num_tokens) *
			shape.intermediate_dim * sizeof(Element)));
	auto tX = make_tensor(
		make_gmem_ptr(in.dX.ptr),
		make_shape(shape.num_tokens, shape.hidden_dim),
		make_stride(shape.hidden_dim, Int<1>{}));
	auto tB = make_tensor(
		make_gmem_ptr(in.dB.ptr),
		make_shape(in.total_n_rows, shape.hidden_dim),
		make_stride(shape.hidden_dim, Int<1>{}));
	auto tZ = make_tensor(
		make_gmem_ptr(dZ),
		make_shape(shape.num_tokens, shape.intermediate_dim),
		make_stride(shape.intermediate_dim, Int<1>{}));
	auto tma_x = make_tma_copy_A_sm100(
		SM100_TMA_2SM_LOAD{}, tX,
		typename Traits::SmemLayoutX_1{},
		typename Traits::TileShape{},
		typename Traits::TiledMma2Sm{});
	auto tma_b = make_tma_copy_B_sm100(
		SM100_TMA_2SM_LOAD{}, tB,
		typename Traits::SmemLayoutW_1{},
		typename Traits::TileShape{},
		typename Traits::TiledMma2Sm{});
	auto tma_z = make_tma_copy(
		SM90_TMA_STORE{}, tZ,
		typename Traits::SmemLayoutStoreSlot{});
	auto kernel = get_kernel<
		decltype(tma_x), decltype(tma_b), decltype(tma_z)>();
	cudaFuncAttributes attributes{};
	CUDA_OK(cudaFuncGetAttributes(&attributes, kernel));
	printf(
		"[mlp1-2sm-spill] regs/thread=%d local=%zu B "
		"static_smem=%zu B\n",
		attributes.numRegs, attributes.localSizeBytes,
		attributes.sharedSizeBytes);
	EXPECT_EQ(attributes.localSizeBytes, 0);
	cudaFree(dZ);
}

TEST(Mlp1Fused2Sm, TFLOPsSquare) {
	if (!blackwell_available())
		GTEST_SKIP() << "requires an sm_100 (Blackwell) GPU";
	if (std::getenv("MLP1_2SM_BENCH") == nullptr)
		GTEST_SKIP() << "set MLP1_2SM_BENCH=1 to benchmark";
	BenchConfig config;
	for (const auto& shape : kSquareBenchShapes)
		run_benchmark(shape, config);
}

TEST(Mlp1Fused2Sm, TFLOPsModels) {
	if (!blackwell_available())
		GTEST_SKIP() << "requires an sm_100 (Blackwell) GPU";
	if (std::getenv("MLP1_2SM_BENCH") == nullptr)
		GTEST_SKIP() << "set MLP1_2SM_BENCH=1 to benchmark";
	BenchConfig config;
	for (const auto& shape : kModelBenchShapes)
		run_benchmark(shape, config);
}

int main(int argc, char** argv) {
	::testing::InitGoogleTest(&argc, argv);
	if (GTEST_FLAG_GET(filter) == "*" &&
		!GTEST_FLAG_GET(list_tests)) {
		std::string filter =
			"Mlp1Fused2Sm.Correctness:"
			"Mlp1Fused2Sm.NoRegisterSpill";
		if (std::getenv("MLP1_2SM_BENCH") != nullptr)
			filter +=
				":Mlp1Fused2Sm.TFLOPsSquare:"
				"Mlp1Fused2Sm.TFLOPsModels";
		GTEST_FLAG_SET(filter, filter);
	}
	return RUN_ALL_TESTS();
}
