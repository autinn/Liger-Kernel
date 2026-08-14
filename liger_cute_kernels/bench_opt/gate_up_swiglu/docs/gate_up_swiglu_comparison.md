# Gate/up + SwiGLU implementation comparison on B200

> **Status: authoritative consolidated report, updated 2026-08-06.**
> The original long-form study is preserved in
> [`archive/gate_up_swiglu_comparison_historical_2026-08-06.md`](archive/gate_up_swiglu_comparison_historical_2026-08-06.md).
> Its old `1.205x` CUDA headline is not an apples-to-apples provider comparison.

This report compares implementations of the MoE MLP phase-1 operation:

```text
gate = X @ W_gate[expert].T
up   = X @ W_up[expert].T
Z    = SiLU(gate) * up
```

The down projection is outside the benchmark scope.

## 1. Executive summary

The current experiment uses one runner, identical deterministic high-entropy
BF16 inputs for every provider, rotating provider order, and pinned standalone
activation implementations.

| Model | CUDA fused | CuTeDSL fused | cuTile fused | Triton fused | cuBLAS + Triton | cuBLAS + CuTeDSL |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B | 1051.3 | 998.2 | 1030.8 | 698.7 | 865.0 | 999.4 |
| Qwen3-235B-A22B | 1027.8 | 976.8 | 1060.5 | 751.8 | 1116.7 | 1229.4 |
| Qwen3.5-122B-A10B | 1054.9 | 986.3 | 1070.0 | 728.3 | 1026.0 | 1148.0 |
| Llama-4-Scout-17B-16E | 1078.9 | 1120.7 | 1143.7 | 836.0 | 1298.4 | 1387.1 |
| Mixtral-8x7B | 1119.7 | 1082.8 | 1144.7 | 839.6 | 1330.3 | 1429.5 |
| Mixtral-8x22B | 1084.0 | 1076.5 | 1125.9 | 856.8 | 1339.4 | 1413.1 |
| **Geomean TFLOP/s** | **1069.1** | **1038.7** | **1095.0** | **782.8** | **1148.3** | **1257.4** |
| **vs CUDA fused** | **1.000x** | **0.972x** | **1.024x** | **0.732x** | **1.074x** | **1.176x** |

For this balanced `M=8192`, `E=8` workload:

1. **cuBLAS + CuTeDSL is fastest**, at `1.176x` the production CUDA fused
   kernel. Equal `M/E` groups let the two projections use efficient batched
   cuBLAS GEMMs, and the PR #1277 CuTeDSL activation is cheaper than the PR
   #1271 Triton activation.
2. **The three Blackwell-native fused implementations are effectively close.**
   CuTeDSL is 2.8% below CUDA and cuTile is 2.4% above it. These small
   differences are shape-dependent and remain exposed to unlocked-clock
   variation.
3. **Triton fused is materially slower.** Its grouped 2-D grid and
   `pre_act[M,2I]` output differ structurally from the TMA/TMEM pipelines used
   by the other fused implementations.
4. **Fusion is not automatically faster.** On this workload, cuBLAS GEMM
   efficiency outweighs the fused kernels' reduction in intermediate traffic.

## 2. Compared providers

| Provider | Kernels | GEMM strategy | Scheduler | Forward outputs |
|---|---:|---|---|---|
| CUDA fused | 1 | CUTLASS C++ `tcgen05` UMMA, 1CTA, `AccStages=2` | Production shape-specific N-split | `Z` |
| CuTeDSL fused | 1 | CuTe DSL `tcgen05` MMA, 1CTA, `AccStages=2` | Fixed 148-block persistent grid | `Z` |
| cuTile fused | 1 | `cuda.tile` MMA, 1CTA | Fixed persistent grid; accumulator staging compiler-managed | `Z` |
| Triton fused | 1 | SonicMoE grouped `tl.dot` | Autotuned grouped 2-D grid | `pre_act`, `Z` |
| cuBLAS + Triton | 3 | Two expert-batched cuBLAS GEMMs | cuBLAS internal | gate, up, `Z` |
| cuBLAS + CuTeDSL | 3 | Two expert-batched cuBLAS GEMMs | cuBLAS internal | gate, up, `Z` |

The standalone activation providers are pinned to:

| Provider | Source revision |
|---|---|
| Triton SwiGLU | Liger-Kernel PR #1271 merge `c5d3e242aff25cd6b121bba9504c76ec66f36412` |
| CuTeDSL SwiGLU | Liger-Kernel PR #1277 head `2199094898dbfef2bc75c1d8b6566baf2ff61695` |

## 3. Benchmark protocol

| Property | Current protocol |
|---|---|
| GPU | 1 NVIDIA B200 (`sm_100`) |
| Shapes | Six model shapes, each with `M=8192`, `E=8`, balanced blocked routing |
| Data | Identical deterministic high-entropy BF16 inputs and weights for all providers |
| Accumulation/output | FP32 accumulation, BF16 output |
| Metric | `4*M*H*I / seconds`; SiLU FLOPs excluded |
| Rounds | Five, with provider order rotated each round |
| Per-round warmup | 200 ms of the provider's own workload plus 10 launches |
| Timing | Median of 50 single-launch CUDA-event samples |
| Python launch handling | CUDA graph replay excludes Python/DLPack setup from device timing |
| Validation | Every provider passes a correctness gate before performance timing |

The deterministic inputs use common seeds for `X`, gate weights, and up weights
in both the Python and CUDA backends. This controls both numerical work and the
physical bit patterns presented to the memory hierarchy.

The experiment and raw output live under:

```text
liger_cute_kernels/bench_opt/gate_up_swiglu/
├── benchmark.py
├── backends/
├── results.csv
└── results_raw.json
```

## 4. Why the previous CUDA result disagreed

The earlier five-way report ranked CUDA fused at `1.205x` the cuBLAS + Triton
baseline. The current matched-input run reverses that ranking.

| Provider | Previous geomean TFLOP/s | Current geomean TFLOP/s | Change |
|---|---:|---:|---:|
| CUDA fused | 1311.8 | 1069.1 | -18.5% |
| cuBLAS + Triton | 1088.9 | 1148.3 | +5.5% |
| cuBLAS + CuTeDSL | 1171.4 | 1257.4 | +7.3% |

### 4.1 The old inputs were not matched

The old Python providers generated random tensors. The old CUDA performance
path instead initialized every byte with:

```cpp
cudaMemset(ptr, 0x3C, count * sizeof(bfloat16));
```

Every two-byte BF16 value was therefore the same `0x3c3c` bit pattern
(approximately `0.0114`). Large buffers containing one repeated pattern are
highly compressible in the B200 memory hierarchy. They can require less
physical L2/HBM traffic and produce a different power and clock operating
point than high-entropy model-like data.

The tensor-core instruction count did not change. The elapsed time changed
because memory-system behavior and power-managed clocks are data-dependent.

### 4.2 Why the repeated pattern was used

The CUDA benchmark originally ran by itself. Generating random host tensors for
the largest model shapes required several gigabytes of host memory and seconds
of setup before each throughput test. `cudaMemset` was a convenient fast path,
and correctness used a separate random-data test.

That shortcut assumed dense GEMM timing was independent of input values. The
assumption does not hold for this B200 wall-clock comparison because
compression and DVFS respond to the physical data pattern.

### 4.3 Interpretation of the old result

This was **not random measurement noise**. It was a deterministic benchmark
confound:

- the old result is reproducible for its exact setup;
- it measures CUDA on repeated compressible data against cuBLAS on random data;
- it does not establish an implementation-level CUDA advantage.

The old CUDA binary was also measured separately from the interleaved Python
providers, while the current runner rotates all providers. The current
standalone activation sources are pinned as well. These changes improve
reproducibility, but the unmatched input bits were the critical
apples-to-apples failure behind the previous CUDA headline.

## 5. How to use these results

Use the current table for the claim:

> On six balanced blocked-routing B200 MoE shapes, cuBLAS + the PR #1277
> CuTeDSL activation reaches `1.176x` the production CUDA fused kernel, while
> the CUDA, CuTeDSL, and cuTile fused implementations are within about 3%
> geomean.

Do not use the historical `1.205x` CUDA-over-cuBLAS number as a general provider
comparison.

The current result is intentionally narrow:

- Routing is perfectly balanced, so two `torch.bmm` calls can represent the
  expert GEMMs. Irregular routing, padding, or small expert groups can change
  the ranking.
- This is forward MLP1 only. It excludes the down projection, backward pass,
  communication, and end-to-end model effects.
- Output contracts differ: the cuBLAS paths materialize gate and up tensors,
  Triton fused writes `pre_act` and `Z`, and the other fused kernels write only
  `Z`.
- This is not a dense `E=1` ceiling. A dedicated dense CUDA kernel could remove
  routing machinery and use different tuning.
- B200 clocks were not locked. Rotating order limits drift, but differences of
  only a few percent should be treated as near parity rather than a universal
  winner.

## 6. Reproduce

```bash
cd liger_cute_kernels/bench_opt/gate_up_swiglu
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
python benchmark.py
```

The runner writes the summary to `results.csv` and the full protocol,
correctness results, commands, round order, and per-round timings to
`results_raw.json`.

## 7. Supporting reports

| Document | Purpose |
|---|---|
| [`gate_up_fused_backend_comparison.md`](gate_up_fused_backend_comparison.md) | Focused CUDA/CuTeDSL/cuTile/Triton fused-provider comparison and generated-code details |
| [`../../BLACKWELL_KERNEL_STACK_COMPARISON.md`](../../BLACKWELL_KERNEL_STACK_COMPARISON.md) | Cross-experiment programming-model analysis using this 1CTA result and the MLP3 2CTA result |
| [`cutile_vs_triton_complexity_assessment.md`](cutile_vs_triton_complexity_assessment.md) | cuTile/Triton evidence across Liger-Kernel PRs and fused SwiGLU |
| [`archive/cutedsl_vs_cuda_forensic.md`](archive/cutedsl_vs_cuda_forensic.md) | Historical CuTeDSL/CUDA profiling and methodology investigation |
| [`archive/matched_persistent_2cta.md`](archive/matched_persistent_2cta.md) | Historical matched persistent-2CTA experiment |
| [`archive/gate_up_swiglu_comparison_historical_2026-08-06.md`](archive/gate_up_swiglu_comparison_historical_2026-08-06.md) | Original long-form report, preserved but superseded |
| [`archive/data/historical_fiveway_moe.json`](archive/data/historical_fiveway_moe.json) | Raw data behind the old MoE headline |
| [`archive/data/historical_fiveway_dense.json`](archive/data/historical_fiveway_dense.json) | Raw data behind the old dense headline |
