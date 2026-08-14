# CUDA C++ vs CuTeDSL vs cuTile: matched persistent-2CTA MLP1

> **Historical 2CTA report.** The current consolidated benchmark now uses the
> production CUDA 1CTA/AccStages=2 pipeline. See
> [`../gate_up_swiglu_comparison.md`](../gate_up_swiglu_comparison.md) for
> current results.

**Date:** 2026-08-06  
**GPU:** 1x NVIDIA B200 (`sm_100`, 148 SMs)  
**Operation:** `Z = SiLU(X @ W_gate[e].T) * (X @ W_up[e].T)`  
**Baseline:** CUDA C++ persistent-2CTA fast math = `1.000x`

## Result

With the output tile, persistent grid, routing, input bits, fast-math path, and
timing protocol matched, all three implementations are at essentially the same
aggregate performance:

| Implementation | Geomean TFLOP/s | Speedup vs CUDA |
|---|---:|---:|
| CUDA C++ | **1084.9** | `1.0000x` |
| CuTeDSL fast math | **1103.6** | **`1.0172x`** |
| cuTile fast math | **1089.0** | **`1.0038x`** |

cuTile is `0.9868x` CuTeDSL in geomean. The per-shape winner changes, and the
largest aggregate difference from CUDA is only 1.7%. Given unlocked B200 clocks
and the visible round-to-round spread, the defensible conclusion is **three-way
performance parity**, not a stable language ranking.

## Per-shape performance

Throughput uses `4*M*H*I/time`; the SiLU FLOPs are not counted.

| Model | CUDA TFLOP/s | CuTeDSL TFLOP/s | DSL/CUDA | cuTile TFLOP/s | cuTile/CUDA |
|---|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B | 1003.2 | 1030.8 | 1.028x | 1016.8 | 1.014x |
| Qwen3-235B-A22B | 1057.6 | 1091.9 | 1.032x | 1080.9 | 1.022x |
| Qwen3.5-122B-A10B | 1024.9 | 1061.9 | 1.036x | 1034.1 | 1.009x |
| Llama-4-Scout-17B-16E | 1135.5 | 1178.7 | 1.038x | 1150.8 | 1.013x |
| Mixtral-8x7B | 1149.4 | 1161.8 | 1.011x | 1151.3 | 1.002x |
| Mixtral-8x22B | 1148.6 | 1103.6 | 0.961x | 1107.6 | 0.964x |
| **Geomean** | **1084.9** | **1103.6** | **1.0172x** | **1089.0** | **1.0038x** |

### Latency and round spread

Each cell is the median of five round medians followed by the min-max range
across those rounds.

| Model | CUDA ms (range) | CuTeDSL ms (range) | cuTile ms (range) |
|---|---:|---:|---:|
| Qwen3-30B-A3B | 0.051376 (0.051248-0.052608) | 0.050000 (0.048560-0.050896) | 0.050688 (0.049904-0.052400) |
| Qwen3-235B-A22B | 0.194928 (0.193408-0.198672) | 0.188800 (0.180608-0.206368) | 0.190736 (0.190432-0.192048) |
| Qwen3.5-122B-A10B | 0.100576 (0.100480-0.100608) | 0.097072 (0.096656-0.099744) | 0.099680 (0.095696-0.104368) |
| Llama-4-Scout-17B-16E | 1.210352 (1.198416-1.227008) | 1.166032 (1.145152-1.266704) | 1.194256 (1.155104-1.222064) |
| Mixtral-8x7B | 1.673984 (1.663024-1.691728) | 1.656160 (1.620416-1.830720) | 1.671280 (1.630320-1.701552) |
| Mixtral-8x22B | 2.871664 (2.857872-2.900016) | 2.988928 (2.862448-3.011600) | 2.978144 (2.811152-3.032320) |

## What was matched

| Property | CUDA C++ | CuTeDSL | cuTile |
|---|---|---|---|
| Forward contract | BF16 `Z` only | BF16 `Z` only | BF16 `Z` only |
| Joined MMA tile | `256x128x64` | `256x128x64` | `256x128x64` |
| CTA mode | true 2CTA | true 2CTA | true 2CTA |
| Physical launch | 148 CTAs / 74 clusters | 148 CTAs / 74 clusters | 148 CTAs / 74 clusters |
| Scheduler | static persistent | static persistent | static persistent |
| Traversal | M-major flattened `(m,n)` | same | same |
| Routing | one expert ID per 256 rows | same | same |
| Input bits | deterministic hash BF16 | same generator/seeds | same generator/seeds |
| Accumulation | two FP32 accumulators | two FP32 accumulators | two FP32 accumulators |
| Math | approximate exp/reciprocal | optional fast-math module | approximate reciprocal |
| Timed operation | one graph replay | one graph replay | one graph replay |

The common traversal is:

```text
linear = cluster_id
while linear < M_tiles * N_tiles:
    m = linear % M_tiles
    n = linear / M_tiles
    process tile (m, n)
    linear += 74
```

All providers use seeds `0x12345678` for `X`, `0x9abcdef0` for gate weights,
and `0x31415926` for up weights.

### Deliberate residual differences

This comparison matches the operation and scheduling, not every compiler-owned
resource decision:

- CUDA uses four explicit TMA stages and 384 threads.
- CuTeDSL selects six TMA stages, uses two epilogue warpgroups, and launches
  320 threads.
- cuTile manages pipeline depth and worker assignment in TileIRAS.
- CUDA was built with nvcc 12.9; CuTeDSL used 4.5.2 with CUDA 13.0; cuTile used
  cuTile 1.5.0 with TileIRAS 13.3.36.

These are part of the implementation each frontend produces. They are reported
below rather than artificially forced to match.

## Correctness

The numerical oracle is FP32 accumulation over the same BF16 inputs. The test
shape is `M=768, H=512, I=256, E=3`, covering multiple M/N tiles and experts.

| Provider | Relative Frobenius | Mean relative | Max relative | Max absolute |
|---|---:|---:|---:|---:|
| CUDA C++ | not emitted | 0.00138148 | 0.00388809 | 0.00382841 |
| CuTeDSL | 0.00166211 | 0.00138162 | 0.00388809 | 0.00382841 |
| cuTile | 0.00166211 | 0.00138162 | 0.00388809 | 0.00382841 |

All are well inside the existing `<1%` mean / `<5%` max-relative gates. The
nearly identical errors also confirm that all three reduce the same gate/up
GEMMs before applying `SiLU(gate) * up`.

## Compiled-code logic check

The comparison uses the fast-math path for all providers:

- CUDA: `--use_fast_math --prec-div=false`
- CuTeDSL: local `FusedSwigluGateUpPersistentKernel` snapshot with an
  `rcp_approx` epilogue
- cuTile: approximate `ct.truediv(..., RoundingMode.APPROX)`

CUDA and CuTeDSL expose PTX. Public cuTile/TileIRAS does **not** expose PTX: it
compiles TileIR bytecode directly to cubin. Therefore the common final check is
SASS, augmented with PTX for CUDA/CuTeDSL and TileIR bytecode for cuTile.

### PTX / TileIR evidence

| Property | CUDA PTX | CuTeDSL PTX | cuTile |
|---|---:|---:|---|
| `tcgen05.mma.cta_group::2` static sites | **8** | **8** | PTX unavailable |
| Approximate reciprocal sites | 128 | 32 | TileIR bytecode + SASS |
| Approximate exponential sites | 128 | 32 | TileIR bytecode + SASS |
| Intermediate artifact | PTX | PTX | 3,435-byte TileIR bytecode |

Eight 2CTA MMA sites represent four K=16 substeps times two destinations:
gate and up. CUDA's later SASS contains cloned/unrolled copies, but its PTX
contains the same eight-operation logical body as CuTeDSL.

### Final SASS and resources

Static instruction counts are not dynamic work counts: nvcc and TileIRAS clone
some pipeline/epilogue paths while CuTeDSL retains compact loops.

| Property | CUDA C++ | CuTeDSL fast | cuTile |
|---|---:|---:|---:|
| `UTCHMMA.2CTA` static sites | 16 | 8 | 8 |
| 2CTA TMA-load static sites | 21 | 3 | 3 |
| TMA-store static sites | 2 | 1 | 2 |
| `MUFU.EX2` static sites | 128 | 32 | 128 |
| `MUFU.RCP` static sites | 130 | 32 | 130 |
| Registers/thread | 167 | 83 | 255 |
| Dynamic/reported shared bytes | 147,584 | 229,632 | 230,748 |
| Stack bytes | 0 | 0 | 0 |
| Local bytes | 0 | 0 | 0 |
| `LDL`/`STL` instructions | 0 | 0 | 0 |

The compiled-code gate passed for all providers:

1. true 2CTA tensor-core instructions are present;
2. both GEMM destinations are represented by eight logical MMA sites;
3. the fast sigmoid path has approximate exponential and reciprocal
   instructions;
4. each kernel has a TMA output path and no `pre_act` output;
5. no provider spills to local memory.

The source signatures and correctness oracle provide the final semantic check:
each kernel accepts only `X`, gate weights, up weights, expert IDs, and `Z`, so
there is no hidden pre-activation output or separate activation launch.

## Timing protocol

- Six MoE shapes, all `M=8192`, `E=8`
- Five rounds per shape
- Provider order rotated each round
- 200 ms provider warmup per round
- 10 launch warmups
- 50 single-replay CUDA-event samples per provider per round
- Reported value: median of five round medians
- Python/DLPack and descriptor construction excluded by CUDA graph replay
- One exclusive GPU lock for the complete campaign

The dataset contains 4,500 timed samples:

```text
6 shapes * 3 providers * 5 rounds * 50 samples = 4,500
```

Clocks could not be locked. Rotating order distributes drift, while the
min-max table exposes the remaining B200 power/DVFS variance.

## Reproduction and artifacts

Build the CUDA baseline:

```bash
cd liger_cute_kernels/bench_opt/gate_up_swiglu
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
```

Run the full comparison:

```bash
python benchmark.py --providers cuda_cpp cutedsl cutile
```

Run the generated-code assertions after producing dumps:

```bash
python benchmark.py --check-codegen --artifact-dir artifacts
```

Archived compact data:

```text
data/matched_2cta_results.csv
```

The generated-instruction and resource summaries are preserved above; binary
compiler artifacts are not duplicated under `docs/`.

## Conclusion

On a matched persistent 2CTA schedule, frontend choice does not materially
change fused gate/up SwiGLU throughput on B200. CuTeDSL fast math is 1.7% above
CUDA in geomean, while cuTile is 0.4% above CUDA; both differences are small
relative to shape-dependent and DVFS variation. Generated PTX/SASS confirms
that the implementations perform the same two-GEMM plus gated-SiLU logic.
