# MLP3 2SM: CUDA/CuTe, CuTe DSL, and cuTile

## Result

All four configurations implement the same isolated Blackwell MLP3 contract:

```text
dA[expert] = dY[expert].T @ Z[expert]
```

## Topline isolated performance

Normalized to CUDA S6:

| Provider | Relative performance | SMEM | Policy |
|---|---:|---:|---|
| **CuTe DSL** | **1.0361x** | 230,400 B | Explicit S6-like |
| CUDA S6 | 1.0000x | 229,632 B | Explicit 6 stages |
| CUDA S5 | 0.9866x | 196,736 B | Explicit 5 stages |
| cuTile | 0.9780x | 230,628 B | Compiler-selected |

CuTe DSL versus CUDA S6:

| Model | CuTe DSL / CUDA S6 |
|---|---:|
| Qwen3-30B-A3B | **1.1333x** |
| Qwen3-235B-A22B | **1.0250x** |
| Qwen3.5-122B-A10B | **1.0826x** |
| Llama-4-Scout-17B-16E | 0.9891x |
| Mixtral-8x7B | 0.9816x |
| Mixtral-8x22B | **1.0129x** |
| **Geomean** | **1.0361x** |

CuTe DSL wins four of six shapes and is **1.0593x cuTile**.

They cover persistent outer-split scheduling, arbitrary per-expert K-block
ranges, empty and skewed experts, split-K, BF16 reduction-add output, and a
logical `256x256x64` two-CTA tile.

On one NVIDIA B200, the repeated six-model campaign found:

| Comparison | Geomean |
|---|---:|
| CuTe DSL / CUDA S5 | **1.0501x** |
| CuTe DSL / CUDA S6 | **1.0361x** |
| CuTe DSL / cuTile | **1.0593x** |
| cuTile / CUDA S5 | 0.9913x |
| cuTile / CUDA S6 | 0.9780x |

An independent repetition produced `1.0503x`, `1.0360x`, and `1.0601x` for
the first three comparisons, respectively.

## Performance

Each latency is the median of five provider-order-rotated rounds.

| Model | CUDA S5 ms | CUDA S6 ms | CuTe DSL ms | cuTile ms | DSL / S6 | DSL / cuTile |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B | 0.034816 | 0.034816 | **0.030720** | 0.032768 | **1.1333x** | **1.0667x** |
| Qwen3-235B-A22B | 0.088064 | 0.084000 | **0.081952** | 0.090128 | **1.0250x** | **1.0998x** |
| Qwen3.5-122B-A10B | 0.053248 | 0.053248 | **0.049184** | 0.053280 | **1.0826x** | **1.0833x** |
| Llama-4-Scout-17B-16E | 0.477184 | **0.468992** | 0.474144 | 0.497664 | 0.9891x | **1.0496x** |
| Mixtral-8x7B | 0.668608 | **0.655424** | 0.667712 | 0.690224 | 0.9816x | **1.0337x** |
| Mixtral-8x22B | 1.118240 | 1.122336 | **1.108000** | 1.135632 | **1.0129x** | **1.0249x** |
| **Geomean** | - | - | - | - | **1.0361x** | **1.0593x** |

CuTe DSL wins four of six shapes. CUDA S6 remains faster on Llama-4-Scout
and Mixtral-8x7B.

## Resource policy

The three deep-pipeline implementations are closely matched:

| Provider | Mainloop stages | Accumulator stages | SMEM budget |
|---|---:|---:|---:|
| CUDA S6 | 6 | 2 | 229,632 B |
| CuTe DSL | 6 | 2 | 230,400 B |
| cuTile | Compiler-selected | Compiler-selected | 230,628 B |

CuTe DSL additionally uses four epilogue SMEM stages, two epilogue
warpgroups, and all 512 TMEM columns. This makes its S6 comparison stronger
than a comparison against the lower-memory S5 integration policy.

## Native-code evidence

Function-specific codegen inspection reports:

| Provider | `UTCHMMA.2CTA` | 2CTA `UTMALDG` | `UTMAREDG.*.ADD` | CGA barriers | Local memory |
|---|---:|---:|---:|---:|---:|
| CUDA S5 | 8 | 4 | 4 | Present | 0 |
| CUDA S6 | 8 | 4 | 4 | Present | 0 |
| CuTe DSL | 4 | 4 | 4 | Present | 0 |
| cuTile | 4 | 4 | 4 | Present | 0 |

CuTe DSL PTX independently contains:

```text
4 x tcgen05.mma.cta_group::2
4 x cp.async.bulk.tensor...cta_group::2
1 x cp.reduce.async.bulk.tensor...add
```

The SASS has four reduction-store sites because the logical PTX store is
partitioned into output subtiles. Static instruction sites are not dynamic
FLOP counts.

## Correctness

- CUDA S5 and S6 each pass 29 existing standalone checks.
- CuTe DSL and cuTile each pass nine matched cases.
- The DSL cases cover balanced inputs, outer split, empty experts, skew,
  prime K-block counts, and split-K values 2 and 3.
- CuTe DSL's worst `k_split=1` mean elementwise relative error is 0.141%.
- Split-K uses absmax normalization because independent BF16 partial sums
  change reduction order.

## Method

| Item | Value |
|---|---|
| GPU | NVIDIA B200, 148 SMs |
| Driver | 580.105.08 |
| CUDA | 12.9.86, `sm_100a` |
| CuTe DSL | `nvidia-cutlass-dsl==4.6.0` |
| cuTile | `cuda-tile==1.5.0`, TileIRAS 13.3.36 |
| Inputs/output | BF16 |
| Tile | `256x256x64`, two CTA |
| Shapes | Six models, `T=8192`, `E=8` |
| Tuning | Independent outer-split sweep |
| Sampling | Five rounds, five warmups and 30 samples per split |
| Cache policy | 256 MiB L2 eviction before every timed launch |
| Timing | CUDA events around only the MLP3 kernel |

## Interpretation

CuTe DSL demonstrates the value of the middle-control programming model:

- it expresses the two-CTA cluster, TMA layouts, TMEM allocation, barriers,
  stage counts, and reduction epilogue explicitly in Python;
- it compiles to the same native instruction classes as CUDA and cuTile;
- it slightly exceeds hand-controlled CUDA S6 in this isolated campaign;
- unlike stable cuTile, it exposes the exact resource policy.

The isolated result does not prove that the 230.4 KiB DSL configuration is
appropriate for the fused backward kernel. Like CUDA S6 and cuTile, it would
compete with the communication bounce buffer. A lower-memory CuTe DSL S5
variant is the relevant next integration experiment.

Raw measurements are in [`results_raw.json`](results_raw.json), compact
results in [`results.csv`](results.csv), and codegen assertions in
[`codegen.json`](codegen.json).
