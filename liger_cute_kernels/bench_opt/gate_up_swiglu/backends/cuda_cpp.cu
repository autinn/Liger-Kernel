// Production-pipelined 1CTA CUDA/CUTLASS MLP1 provider.
//
// Configuration: TileM/N/K=128/128/64, four TMA stages, AccStages=2,
// two epilogue warpgroups, and the production shape-specific N-split schedule.

#include <chrono>

#include "cuda_1sm_support.cuh"

namespace gate_up_swiglu {

using namespace gate_up_swiglu_cuda;

using Traits = Traits1Sm;
using BenchShape = gate_up_swiglu_cuda::Shape;

constexpr uint32_t kInputSeed = 0x12345678U;
constexpr uint32_t kGateSeed = 0x9abcdef0U;
constexpr uint32_t kUpSeed = 0x31415926U;

static const std::vector<BenchShape>& comparison_shapes() {
	static const std::vector<BenchShape> shapes = {
		{"Qwen3-30B-A3B", 8192, 2048, 768, 8},
		{"Qwen3-235B-A22B", 8192, 4096, 1536, 8},
		{"Qwen3.5-122B-A10B", 8192, 3072, 1024, 8},
		{"Llama-4-Scout-17B-16E", 8192, 5120, 8192, 8},
		{"Mixtral-8x7B", 8192, 4096, 14336, 8},
		{"Mixtral-8x22B", 8192, 6144, 16384, 8},
	};
	return shapes;
}

static const BenchShape& find_shape(const std::string& name) {
	for (const auto& shape : comparison_shapes())
		if (name == shape.name)
			return shape;
	std::fprintf(stderr, "unknown model: %s\n", name.c_str());
	std::exit(2);
}

static int production_split(const std::string& name) {
	if (name == "Qwen3-30B-A3B" ||
	    name == "Qwen3-235B-A22B" ||
	    name == "Qwen3.5-122B-A10B")
		return 2;
	if (name == "Llama-4-Scout-17B-16E" ||
	    name == "Mixtral-8x7B")
		return 16;
	if (name == "Mixtral-8x22B")
		return 32;
	std::fprintf(stderr, "no production split for model: %s\n", name.c_str());
	std::exit(2);
}

static DeviceInputs make_comparison_inputs(const BenchShape& shape) {
	auto inputs = make_benchmark_inputs(shape);
	size_t x_count =
		static_cast<size_t>(shape.tokens) * shape.hidden;
	size_t w_count =
		static_cast<size_t>(shape.experts) *
		shape.intermediate *
		shape.hidden;
	fill_bf16_kernel<<<4096, 256>>>(
		inputs.x, x_count, kInputSeed, false);
	fill_bf16_kernel<<<4096, 256>>>(
		inputs.b, w_count, kGateSeed, false);
	fill_bf16_kernel<<<4096, 256>>>(
		inputs.c, w_count, kUpSeed, false);
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaDeviceSynchronize());
	return inputs;
}

static void run_correctness() {
	BenchShape shape{"correctness", 768, 512, 256, 3};
	auto inputs = make_comparison_inputs(shape);
	size_t output_count =
		static_cast<size_t>(inputs.padded_tokens_1sm) *
		shape.intermediate;
	Element* output = nullptr;
	CUDA_CHECK(cudaMalloc(
		&output, output_count * sizeof(Element)));
	CUDA_CHECK(cudaMemset(
		output, 0, output_count * sizeof(Element)));

	auto tma_x = make_one_sm_x_tma<Traits>(inputs.x, shape);
	auto tma_b = make_one_sm_w_tma<Traits>(inputs.b, shape);
	auto tma_c = make_one_sm_w_tma<Traits>(inputs.c, shape);
	auto tma_z = make_one_sm_store_tma<Traits>(
		output, inputs.padded_tokens_1sm, shape);
	set_one_sm_fused_attributes(tma_x, tma_b, tma_z);
	launch_one_sm_fused(
		tma_x,
		tma_b,
		tma_c,
		tma_z,
		inputs,
		shape,
		2,
		inputs.m_tiles_1sm,
		0);
	CUDA_CHECK(cudaDeviceSynchronize());

	HostInputs host;
	host.x = download(
		inputs.x,
		static_cast<size_t>(shape.tokens) *
			shape.hidden);
	host.b = download(
		inputs.b,
		static_cast<size_t>(shape.experts) *
			shape.intermediate *
			shape.hidden);
	host.c = download(
		inputs.c,
		static_cast<size_t>(shape.experts) *
			shape.intermediate *
			shape.hidden);
	auto expected =
		cpu_reference(shape, host, inputs, false);
	auto actual = download(output, output_count);
	auto error = compare(actual, expected.z);
	std::printf(
		"CORRECTNESS,provider=cuda,"
		"mean_relative=%.9g,max_relative=%.9g,"
		"max_absolute=%.9g\n",
		error.mean_relative,
		error.max_relative,
		error.max_absolute);
	if (error.mean_relative >= 0.01 ||
	    error.max_relative >= 0.05) {
		std::fprintf(stderr, "correctness gate failed\n");
		std::exit(3);
	}
	CUDA_CHECK(cudaFree(output));
}

template <typename Replay>
static void warm_provider(
		Replay replay,
		double warmup_ms,
		int launch_warmups,
		cudaStream_t stream) {
	auto begin = std::chrono::steady_clock::now();
	while (std::chrono::duration<double, std::milli>(
			std::chrono::steady_clock::now() - begin).count() <
			warmup_ms) {
		for (int i = 0; i < 10; ++i)
			replay();
		CUDA_CHECK(cudaStreamSynchronize(stream));
	}
	for (int i = 0; i < launch_warmups; ++i)
		replay();
	CUDA_CHECK(cudaStreamSynchronize(stream));
}

template <typename Replay>
static std::vector<float> sample_provider(
		Replay replay,
		int samples,
		cudaStream_t stream) {
	cudaEvent_t start, stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));
	std::vector<float> timings;
	timings.reserve(samples);
	for (int i = 0; i < samples; ++i) {
		CUDA_CHECK(cudaEventRecord(start, stream));
		replay();
		CUDA_CHECK(cudaEventRecord(stop, stream));
		CUDA_CHECK(cudaEventSynchronize(stop));
		float milliseconds = 0.0f;
		CUDA_CHECK(cudaEventElapsedTime(
			&milliseconds, start, stop));
		timings.push_back(milliseconds);
	}
	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));
	return timings;
}

static void run_benchmark(
		const BenchShape& shape,
		int rounds,
		int samples,
		double warmup_ms,
		int launch_warmups) {
	auto inputs = make_comparison_inputs(shape);
	size_t output_count =
		static_cast<size_t>(inputs.padded_tokens_1sm) *
		shape.intermediate;
	Element* output = nullptr;
	CUDA_CHECK(cudaMalloc(
		&output, output_count * sizeof(Element)));

	auto tma_x = make_one_sm_x_tma<Traits>(inputs.x, shape);
	auto tma_b = make_one_sm_w_tma<Traits>(inputs.b, shape);
	auto tma_c = make_one_sm_w_tma<Traits>(inputs.c, shape);
	auto tma_z = make_one_sm_store_tma<Traits>(
		output, inputs.padded_tokens_1sm, shape);
	set_one_sm_fused_attributes(tma_x, tma_b, tma_z);
	const int split = production_split(shape.name);

	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));
	CUDA_CHECK(cudaStreamBeginCapture(
		stream, cudaStreamCaptureModeGlobal));
	launch_one_sm_fused(
		tma_x,
		tma_b,
		tma_c,
		tma_z,
		inputs,
		shape,
		split,
		inputs.m_tiles_1sm,
		stream);
	cudaGraph_t graph;
	CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
	cudaGraphExec_t graph_exec;
	CUDA_CHECK(cudaGraphInstantiate(
		&graph_exec, graph, 0));
	auto replay = [&]() {
		CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
	};

	std::vector<float> round_medians;
	round_medians.reserve(rounds);
	for (int round = 0; round < rounds; ++round) {
		warm_provider(
			replay, warmup_ms,
			launch_warmups, stream);
		auto timings =
			sample_provider(replay, samples, stream);
		float round_median =
			static_cast<float>(median(timings));
		round_medians.push_back(round_median);
		std::printf(
			"ROUND,provider=cuda,model=%s,round=%d,"
			"median_ms=%.6f\n",
			shape.name, round + 1, round_median);
	}
	double milliseconds = median(round_medians);
	double throughput = tflops_of(shape, milliseconds);
	auto [min_it, max_it] = std::minmax_element(
		round_medians.begin(), round_medians.end());
	std::printf(
		"RESULT,provider=cuda,model=%s,latency_ms=%.6f,"
		"tflops=%.4f,round_min_ms=%.6f,"
		"round_max_ms=%.6f,split=%d\n",
		shape.name,
		milliseconds,
		throughput,
		*min_it,
		*max_it,
		split);

	CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
	CUDA_CHECK(cudaGraphDestroy(graph));
	CUDA_CHECK(cudaStreamDestroy(stream));
	CUDA_CHECK(cudaFree(output));
}

}  // namespace gate_up_swiglu

int main(int argc, char** argv) {
	std::string model = "Qwen3-30B-A3B";
	int rounds = 1;
	int samples = 50;
	int launch_warmups = 10;
	double warmup_ms = 200.0;
	bool correctness_only = false;
	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if (arg == "--model" && i + 1 < argc)
			model = argv[++i];
		else if (arg == "--rounds" && i + 1 < argc)
			rounds = std::atoi(argv[++i]);
		else if (arg == "--samples" && i + 1 < argc)
			samples = std::atoi(argv[++i]);
		else if (arg == "--warmup-ms" && i + 1 < argc)
			warmup_ms = std::atof(argv[++i]);
		else if (arg == "--launch-warmups" &&
		         i + 1 < argc)
			launch_warmups = std::atoi(argv[++i]);
		else if (arg == "--correctness-only")
			correctness_only = true;
		else {
			std::fprintf(
				stderr, "unknown argument: %s\n",
				arg.c_str());
			return 2;
		}
	}

	std::printf(
		"CONFIG,provider=cuda,cta_mode=1,"
		"stages=%d,acc_stages=%d,dynamic_smem_bytes=%zu,"
		"threads=%d\n",
		gate_up_swiglu::Traits::Stages,
		gate_up_swiglu::Traits::AccStages,
		sizeof(gate_up_swiglu_cuda::OneSmFusedKernelSmem<
			gate_up_swiglu::Traits>),
		gate_up_swiglu::Traits::NumThreads);
	if (correctness_only) {
		gate_up_swiglu::run_correctness();
		return 0;
	}
	gate_up_swiglu::run_benchmark(
		gate_up_swiglu::find_shape(model),
		rounds,
		samples,
		warmup_ms,
		launch_warmups);
	return 0;
}
