# LigerCute Blackwell Port and Optimization Report

## Executive summary

This report separates the Blackwell work into a baseline and three named
optimizations:

1. A correct SM100 baseline using Blackwell UMMA (`tcgen05`) and TMEM.
2. Warp specialization to separate communication, UMMA issue, and epilogue
   work.
3. TMEM accumulator pipelining to overlap the next UMMA output with the
   current epilogue, including the backward MLPs.
4. SMEM mainloop pipelining and paired-CTA 2SM execution for backward MLP3 and
   MLP4.

The final implementation (`761d687`) is **1.011x faster in forward, 1.586x
faster in backward, and 1.470x faster in forward+backward** than the stable
single-B200 Blackwell baseline (`db3ed8`). The forward path is effectively
preserved, while the backward path receives most of the benefit.

On the 8,192-token named-model matrix, final B200 LCK is **1.776x faster in
forward, 2.108x faster in backward, and 2.028x faster in
forward+backward** than H200 LCK. On B200, it is **2.220x faster
forward+backward than Transformer Engine** and **1.751x faster than DeepEP**.

| Stage | Controlled comparison | Forward | Backward | Forward + backward |
|---|---|---:|---:|---:|
| Blackwell baseline | Reference on 1 B200 | 1.000x | 1.000x | 1.000x |
| Warp specialization | Pre-specialization -> specialized roles, 1 B200 | 1.008x | Not measured | Not measured |
| Accumulator/backward pipeline | Blackwell baseline -> deployed pipeline, 1 B200 | 1.006x | **1.493x** | **1.401x** |
| Mainloop pipeline + paired 2SM | Pre-2SM pipeline -> final, 1 B200 | 1.004x | **1.062x** | **1.049x** |
| Mainloop pipeline + paired 2SM | Pre-2SM pipeline -> final, 8 B200 | 1.002x | **1.127x** | **1.096x** |
| Overall software impact | Blackwell baseline -> final, 1 B200 | **1.011x** | **1.586x** | **1.470x** |
| Final B200 vs H200 LCK | 28 matched 1/2/4/8-GPU points | **1.776x** | **2.108x** | **2.028x** |
| Final B200 LCK vs Transformer Engine | 28 matched B200 points | **1.800x** | **2.330x** | **2.220x** |
| Final B200 LCK vs DeepEP | 21 matched 2/4/8-GPU B200 points | **2.272x** | **1.561x** | **1.751x** |

The optimization rows are controlled comparisons against the immediately
named implementation, not additive contributions. In particular, the
accumulator result includes the warp-role changes and the integration of the
pipeline across the fused backward kernel. The hardware and backend rows use
the separate named-model matrix and are not software-only comparisons.

## Measurement method and checkpoint names

The historical optimization matrix contains seven named MoE shapes at five
token counts (1K, 2K, 4K, 8K, and 16K), for 35 cases. Runs use BF16,
`N_MICROBATCHES=1`, and correctness checking.

Speedup is:

```text
speedup = before_latency / after_latency
```

Forward+backward speedup is calculated for each shape before taking the
geometric mean:

```text
combined_speedup =
    (before_forward_ms + before_backward_ms)
    / (after_forward_ms + after_backward_ms)
```

It is not an average of the forward and backward speedups.

The report uses descriptive checkpoint names rather than letter aliases:

| Checkpoint | Revision | Purpose |
|---|---|---|
| Stable Blackwell baseline | `db3ed8` | Correct single-B200 SM100 implementation with one TMEM accumulator stage |
| Two accumulator stages before role specialization | `448cbbe` | Control point with `AccStages=2` |
| Warp-specialized forward path | `a52b5f` | Dedicated UMMA producer and two epilogue warpgroups |
| Deployed accumulator/backward pipeline | `69cfed` | Two-stage TMEM pipeline across forward and backward, with SM100 tuning tables |
| Final mainloop/2SM implementation | `761d687` | Paired-CTA MLP3/MLP4 and shape-selected SMEM stage depth |

The initial Blackwell port merged in PR
[#1355](https://github.com/linkedin/Liger-Kernel/pull/1355) as `16e49d4`.
The stable baseline follows that port and includes the fixes and single-GPU
tuning needed for a reproducible pre-pipeline comparison. The paired 2SM work
is PR [#1365](https://github.com/linkedin/Liger-Kernel/pull/1365).

## 1. Baseline: correct Blackwell implementation

### What the baseline does

LigerCute is a persistent fused expert-parallel MoE kernel. Dedicated NVSHMEM
warps move remote token tiles through a symmetric-memory ring while the
remaining warps execute the expert MLP. Forward fuses dispatch, SwiGLU
MLP1/MLP2, and combine. Backward fuses gradient communication with activation
backward, MLP2-T, MLP3, MLP4, and MLP5:

```text
MLP1:   U = X B, V = X C, Z = SiLU(U) * V
MLP2:   Y = Z A^T
MLP2-T: dZ = dY A
MLP3:   dA = dY^T Z
MLP4:   dB = dU^T X, dC = dV^T X
MLP5:   dX = dU B + dV C
```

The Blackwell port preserves this fused structure but replaces Hopper WGMMA
and register accumulators with SM100 UMMA and TMEM. Compile-time
specializations keep the architectures separate:

```text
Compute=90  -> Hopper WGMMA implementation
Compute=100 -> Blackwell UMMA/TMEM implementation
```

The stable baseline (`db3ed8`) is the reference before accumulator and
mainloop pipelining. It uses one TMEM accumulator stage and runs on one B200.

### Baseline performance

The baseline's geometric-mean latencies across the 35 cases are:

| Hardware | Forward | Backward | Forward + backward |
|---|---:|---:|---:|
| 1 B200 | 1.962 ms | 11.918 ms | 13.924 ms |

These values define 1.000x for the cumulative software comparison. There is no
equivalent historical 8-B200 baseline: this checkpoint has no valid 8-GPU
tuning table. The report therefore does not manufacture an 8-GPU cumulative
speedup from unmatched configurations.

## 2. Warp specialization

### What it does

The original Blackwell schedule did not balance UMMA issue and epilogue work
well. The controlled comparison keeps two TMEM accumulator stages fixed and
changes only the warp-role package.

Before specialization (`448cbbe`):

```text
WG0: W0 TMA | W1 GET | W2 GET | W3 PUT
WG1: W4 UMMA-only | W5-W7 idle
WG2: W8-W11 epilogue
```

After specialization (`a52b5f`):

```text
WG0: W0 TMA | W1 GET | W2 PUT | W3 UMMA-only
WG1: W4-W7 epilogue
WG2: W8-W11 epilogue
```

Communication is consolidated onto two warps. Warp 3 becomes a dedicated,
epilogue-free UMMA producer, while warps 4-11 form two complete epilogue
warpgroups. The producer can issue work for the next TMEM stage while both
consumer warpgroups drain the current stage. Barrier participation changes
with the new roles.

### Performance

| Hardware and scope | Forward | Backward | Forward + backward |
|---|---:|---:|---:|
| 1 B200, 35-case controlled comparison | **1.008x** | Not measured | Not measured |

The forward geomean is a modest 0.8% improvement: 16 wins, 18 parity cases,
and one regression, with a 0.975x-1.120x range. This stage primarily changes
forward MLP1. The matched backward sweep completed only 2 of 35 cases, so no
backward or combined result is claimed.

The main value of warp specialization is structural rather than its isolated
forward speedup: it creates independent producer and consumer roles that make
the two-stage accumulator pipeline useful.

## 3. TMEM accumulator pipelining

### What it does

The baseline has one TMEM accumulator bank. UMMA writes a result, then waits
for the epilogue to drain that storage before reusing it. The optimization
introduces `PipelineUmmaAsync<2>` and two TMEM banks:

```text
time       t0             t1             t2
UMMA       N -> bank 0    N+1 -> bank 1  N+2 -> bank 0
epilogue                  drain bank 0   drain bank 1
```

The UMMA warp uses `producer_acquire` and `producer_commit`; the epilogue
warpgroups use `consumer_wait` and `consumer_release`. This is an output-side
pipeline in TMEM, separate from operand prefetch in SMEM and from the NVSHMEM
communication ring.

The same producer/consumer separation is propagated through backward MLP2-T,
MLP3, MLP4, and MLP5. Warp 3 issues the next UMMA K-loop while warps 4-11
drain the previous result, process it in `EpiChunkN` pieces, and overlap TMA
stores or `TMA_REDUCE_ADD`. This removes serialization around the
weight-gradient epilogues and reduction/store paths.

### Optimization impact on 1 B200

The clean end-to-end comparison is the stable Blackwell baseline (`db3ed8`)
against the deployed accumulator/backward pipeline (`69cfed`):

| Forward | Backward | Forward + backward |
|---:|---:|---:|
| 1.006x | **1.493x** | **1.401x** |

This is a 0.6% forward improvement, a 33.0% backward latency reduction, and a
28.6% combined latency reduction. All 35 backward and combined cases improve.

This result measures the complete deployed package: two TMEM accumulator
stages, warp specialization, forward MLP2 integration, backward epilogue
pipelining, and the associated SM100 tuning tables. It is not an isolated
microbenchmark of changing `AccStages` from one to two.

### Deployed performance on 1 and 8 B200s

The deployed accumulator checkpoint has complete 35-case measurements on both
one and eight B200s:

| Hardware | Forward geomean | Backward geomean | Forward + backward geomean |
|---|---:|---:|---:|
| 1 B200 | 1.951 ms | 7.981 ms | 9.940 ms |
| 8 B200 | 2.091 ms | 6.599 ms | 8.710 ms |

The token count is per rank, so the one- and eight-B200 rows are deployment
snapshots, not a strong-scaling ratio. A controlled 8-B200 accumulator
speedup is not claimed because the pre-pipeline checkpoint lacks a valid
8-GPU tuning table; only the deployed 8-B200 latency is reported.

## 4. SMEM mainloop pipelining and paired-CTA 2SM

### What it does

Mainloop pipelining and paired-CTA execution address different bottlenecks
from the TMEM accumulator pipeline:

| Mechanism | Storage or resource | Purpose |
|---|---|---|
| SMEM mainloop stages (`Stages3`) | Per-CTA SMEM | Prefetch future K tiles while UMMA consumes the current tile |
| TMEM accumulator stages (`AccStages`) | TMEM | Run UMMA for output N+1 while the epilogue drains output N |
| Paired-CTA 2SM | Two clustered CTAs | Cooperatively compute one larger logical output tile |
| Communication stages | Symmetric HBM ring | Overlap remote token movement with expert compute |

Backward MLP3 and MLP4 launch a two-CTA cluster on SM100:

```text
clusterDim = (2, 1, 1)
    CTA 0 / SM 0 -> output rows 0-127
    CTA 1 / SM 1 -> output rows 128-255
```

Pair-aware TMA descriptors and multicast masks populate each CTA's private
SMEM stages. `Allocator2Sm` coordinates paired TMEM allocation, and a
`cta_group::2` UMMA atom computes a joined 256x256 tile. Each CTA drains its
own 128-row half. Cluster synchronization protects allocation, execution,
reuse, and release.

Within each CTA, the SMEM mainloop is a circular queue. TMA fills a future K
tile while paired UMMA consumes the current tile. Offline tuning selects four
or five SMEM stages by shape; the two-stage TMEM accumulator depth remains
independent.

### Performance

The comparison is the deployed accumulator pipeline (`69cfed`) against the
final mainloop/2SM implementation (`761d687`).

| Hardware | Forward | Backward | Forward + backward |
|---|---:|---:|---:|
| 1 B200 | 1.004x | **1.062x** | **1.049x** |
| 8 B200 | 1.002x | **1.127x** | **1.096x** |

Detailed distributions:

| Hardware and pass | Median | Range | Wins / parity / regressions |
|---|---:|---:|---:|
| 1 B200 forward | 1.002x | 0.917x-1.126x | 12 / 21 / 2 |
| 1 B200 backward | 1.044x | 0.997x-1.183x | 33 / 2 / 0 |
| 1 B200 forward + backward | 1.035x | 0.997x-1.140x | 33 / 2 / 0 |
| 8 B200 forward | 1.000x | 0.954x-1.158x | 8 / 19 / 8 |
| 8 B200 backward | 1.139x | 1.013x-1.282x | 35 / 0 / 0 |
| 8 B200 forward + backward | 1.103x | 1.007x-1.211x | 35 / 0 / 0 |

Forward is nearly unchanged because paired 2SM targets backward MLP3/MLP4.
The eight-B200 backward gain is larger, and every eight-B200 backward and
combined case improves.

The result combines paired-CTA execution with shape-selected mainloop depth.
It does not isolate cluster launch, the 2SM UMMA atom, or stage five as
independent effects.

## 5. Overall impact against the Blackwell baseline

The cumulative software-only comparison uses the same 35 cases on one B200:
stable Blackwell baseline (`db3ed8`) against the final implementation
(`761d687`).

| Pass | Speedup | Latency reduction | Wins / parity / regressions |
|---|---:|---:|---:|
| Forward | **1.011x** | 1.1% | 20 / 1 / 14 |
| Backward | **1.586x** | 36.9% | 35 / 0 / 0 |
| Forward + backward | **1.470x** | 32.0% | 35 / 0 / 0 |

The forward path remains effectively flat while the backward path is
substantially faster. Every combined case improves, with per-shape speedups
from 1.174x to 1.945x. The 1.470x combined result is therefore broad across
the matrix rather than driven by a few outliers.

## 6. Final B200 compared with H200

At 8,192 tokens per rank, the final B200 LCK implementation is compared with
H200 LCK over seven models and GPU counts 1, 2, 4, and 8, for 28 matched
points. Each GPU row is the geometric mean across the seven models, and the
overall row is the geometric mean across all 28 points:

| GPU count | Forward | Backward | Forward + backward |
|---:|---:|---:|---:|
| 1 | 1.681x | 2.084x | 1.994x |
| 2 | 1.815x | 2.174x | 2.086x |
| 4 | 1.839x | 2.163x | 2.080x |
| 8 | 1.773x | 2.014x | 1.954x |
| **Overall** | **1.776x** | **2.108x** | **2.028x** |

This comparison includes both the optimized Blackwell software path and the
B200-versus-H200 hardware change. It is not an isolated software optimization
result. It shows that the final Blackwell implementation converts the newer
hardware into roughly a 2x combined latency advantage across every GPU count.
All 28 matched forward, backward, and combined configurations are faster on
B200.

## 7. Final B200 compared with other B200 backends

The same named-model matrix compares final LCK with Transformer Engine (TE)
and DeepEP on B200. TE has data for all 28 configurations. DeepEP has no
single-GPU measurements, so its aggregate covers the 21 matched 2-, 4-, and
8-GPU configurations.

LCK speedup over Transformer Engine:

| GPU count | Forward | Backward | Forward + backward |
|---:|---:|---:|---:|
| 1 | 2.041x | 3.422x | 3.154x |
| 2 | 1.915x | 2.490x | 2.363x |
| 4 | 1.717x | 2.049x | 1.968x |
| 8 | 1.564x | 1.687x | 1.656x |
| **Overall** | **1.800x** | **2.330x** | **2.220x** |

LCK speedup over DeepEP:

| GPU count | Forward | Backward | Forward + backward |
|---:|---:|---:|---:|
| 2 | 2.463x | 1.701x | 1.890x |
| 4 | 2.253x | 1.567x | 1.751x |
| 8 | 2.114x | 1.427x | 1.621x |
| **Overall** | **2.272x** | **1.561x** | **1.751x** |

LCK wins combined latency in all 28 comparisons with TE and all 21
comparisons with DeepEP. Against TE, it also wins 27 of 28 forward points and
27 of 28 backward points. Against DeepEP, it wins every matched forward and
backward point.

## Overall impact

The optimization sequence changes how work overlaps rather than changing the
fused MoE algorithm:

- Warp specialization separates communication, UMMA production, and
  epilogue consumption.
- TMEM accumulator pipelining turns those roles into temporal overlap and
  removes backward epilogue serialization.
- SMEM mainloop pipelining overlaps operand movement with UMMA.
- Paired-CTA 2SM execution increases the spatial granularity of backward
  MLP3/MLP4.

Together, these changes deliver **1.470x forward+backward speedup over the
stable Blackwell software baseline** and **2.028x over H200 LCK** on the
matched hardware comparison. On B200, final LCK is **2.220x faster
forward+backward than Transformer Engine** and **1.751x faster than DeepEP**.
The primary software gain is backward: **1.586x over the Blackwell baseline**,
while forward remains at **1.011x**.

## Sources

- `blackwell_lck_optimization_strategy.md`
- `blackwell_moe_optimization_comparison.md`
- `liger_cute_kernels/README.md`
- PR [#1355](https://github.com/linkedin/Liger-Kernel/pull/1355)
- PR [#1365](https://github.com/linkedin/Liger-Kernel/pull/1365)
- Historical 35-case checkpoint matrix, including the one- and eight-B200
  deployed accumulator measurements:
  `/home/jobuser/.copilot/session-state/c92f3156-8fbd-4bcd-8c77-8a452227c705/files/blackwell-detached-matrix`
- Final 2-/4-GPU B200 measurements:
  `/home/jobuser/.copilot/session-state/22abb3a8-c60a-43ba-9d4d-ad13015b4785/files/lck-b200-2sm-2gpu-4gpu`
