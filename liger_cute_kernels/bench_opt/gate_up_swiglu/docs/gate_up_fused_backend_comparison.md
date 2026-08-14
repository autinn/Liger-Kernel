# Gate/up fused backend comparison: CUDA, CuTeDSL, cuTile, and Triton

All four providers fuse the two expert GEMMs and SwiGLU into one GPU kernel:

```text
Z = SiLU(X @ W_gate[expert].T) * (X @ W_up[expert].T)
```

## Performance

TFLOP/s; higher is better.

| Model | CUDA C++ | CuTeDSL | cuTile | Triton |
|---|---:|---:|---:|---:|
| Qwen3-30B-A3B | 1051.3 | 998.2 | 1030.8 | 698.7 |
| Qwen3-235B-A22B | 1027.8 | 976.8 | 1060.5 | 751.8 |
| Qwen3.5-122B-A10B | 1054.9 | 986.3 | 1070.0 | 728.3 |
| Llama-4-Scout-17B-16E | 1078.9 | 1120.7 | 1143.7 | 836.0 |
| Mixtral-8x7B | 1119.7 | 1082.8 | 1144.7 | 839.6 |
| Mixtral-8x22B | 1084.0 | 1076.5 | 1125.9 | 856.8 |
| **Geomean** | **1069.1** | **1038.7** | **1095.0** | **782.8** |
| **vs CUDA** | **1.000x** | **0.972x** | **1.024x** | **0.732x** |

| Finding | Result |
|---|---|
| Blackwell-native spread | CUDA, CuTeDSL, and cuTile are within 3% |
| Fastest fused provider | cuTile, 2.4% above CUDA |
| Triton gap | 26.8% below CUDA; 28.5% below cuTile |

## Configuration

| Property | CUDA C++ | CuTeDSL | cuTile | Triton |
|---|---|---|---|---|
| CTA mode | 1CTA | 1CTA | 1CTA | Triton CTA |
| MMA tile | `128x128x64` | `128x128x64` | `128x128x64` | Autotuned |
| Accumulator stages | 2 explicit | 2 explicit | TileIRAS-managed | Compiler-managed |
| Input pipeline | 4 TMA stages | 4 TMA stages | TileIRAS-managed | `num_stages` autotuned |
| Scheduler | Shape-specific N-split | 148-block persistent | 148-block persistent | 2-D grouped grid |
| Global outputs | `Z` | `Z` | `Z` | `pre_act`, `Z` |

CUDA uses N-splits `2,2,2,16,16,32` for the six rows. cuTile does not expose
pipeline or accumulator-depth controls, so its staging cannot be called an
explicit `AccStages=2` match.

## Implementation

| Provider | Core design |
|---|---|
| CUDA C++ | Reuses one X fragment for gate/up; double-buffers two TMEM gate/up accumulator sets; loops over N tiles using a tuned split |
| CuTeDSL | Same 1CTA tile and explicit `num_acc_stage=2`; fixed persistent scheduler; two epilogue warpgroups |
| cuTile | Uses `ct.load` and two `ct.mma` calls per K tile; fixed persistent scheduler; TileIRAS chooses workers and staging |
| Triton | Uses two `tl.dot` calls per K tile; one program per `(M,N)` tile; stores pre-activations for backward |

CUDA and CuTeDSL explicitly overlap MMA for tile `n+1` with the epilogue for
tile `n`. Triton input `num_stages` is not equivalent to explicit TMEM
accumulator double-buffering.

Removing Triton's `pre_act` stores historically improved performance only
slightly. Its larger gap therefore comes mainly from the grouped-GEMM pipeline,
scheduling, and locality rather than output traffic alone.

## Generated code

| Property | CUDA | CuTeDSL | cuTile |
|---|---:|---:|---:|
| PTX 1CTA `tcgen05.mma` sites | 8 | 8 | PTX unavailable |
| SASS 1CTA `UTCHMMA` sites | 16 | 8 | 8 |
| SASS `.2CTA` sites | 0 | 0 | 0 |
| TMA-load sites | 21 | 3 | 3 |
| TMA-store sites | 2 | 1 | 2 |
| Registers/thread | 163 | 83 | 255 |
| Stack/local bytes | 0/0 | 0/0 | 0/0 |

Eight logical MMA sites equal four K=16 substeps for each of gate and up.
CUDA's 16 SASS occurrences are cloned/unrolled control-flow bodies, not twice
the dynamic MMA work.

cuTile exposes TileIR bytecode and cubin/SASS, but not PTX. All three checks
confirm 1CTA tensor-core execution, both GEMM destinations, TMA traffic, and no
local-memory spills.

## Interpretation

1. CUDA, CuTeDSL, and cuTile are at practical geomean parity.
2. Triton's kernel design is less effective on these B200 grouped shapes.
3. This is not a frontend-only comparison: CUDA uses tuned N-splits while the
   two DSL providers use fixed persistent scheduling.

Broader results, including unfused cuBLAS providers, are in
[`gate_up_swiglu_comparison.md`](gate_up_swiglu_comparison.md). Raw data:
[`../results.csv`](../results.csv), [`../results_raw.json`](../results_raw.json),
and [`../codegen.json`](../codegen.json).
