# MLP3 2SM in CUDA/CuTe, CuTe DSL, and cuTile

## Executive summary

This experiment implements the same isolated Blackwell MLP3 weight-gradient
operation in three programming models:

```text
dA[expert] = dY[expert].T @ Z[expert]
```

### Topline isolated 2SM performance

Normalized to the resource-matched CUDA S6 implementation:

| Provider | Relative performance | SMEM | Policy |
|---|---:|---:|---|
| **CuTe DSL** | **1.0361x** | 230,400 B | Explicit S6-like |
| CUDA S6 | 1.0000x | 229,632 B | Explicit 6 stages |
| CUDA S5 | 0.9866x | 196,736 B | Explicit 5 stages |
| cuTile | 0.9780x | 230,628 B | Compiler-selected |

CuTe DSL versus CUDA S6 by model:

| Model | CuTe DSL / CUDA S6 |
|---|---:|
| Qwen3-30B-A3B | **1.1333x** |
| Qwen3-235B-A22B | **1.0250x** |
| Qwen3.5-122B-A10B | **1.0826x** |
| Llama-4-Scout-17B-16E | 0.9891x |
| Mixtral-8x7B | 0.9816x |
| Mixtral-8x22B | **1.0129x** |
| **Geomean** | **1.0361x** |

CuTe DSL wins four of six shapes and is `1.0593x` cuTile in geomean. An
independent full-campaign repeat produced `1.0360x` CUDA S6 and `1.0601x`
cuTile.

- The CUDA/CuTe implementation explicitly programs CTA rank, cluster launch,
  pair-aware TMA, SMEM pipelines, TMEM allocation, `cta_group::2` UMMA,
  accumulator double buffering, and TMA reduction stores.
- The CuTe DSL implementation expresses the same mechanisms explicitly in
  Python using `CtaGroup.TWO`, `PipelineTmaUmma`, `PipelineUmmaAsync`,
  `TmemAllocator`, and a TMA reduction copy atom.
- The cuTile implementation describes a logical tile program and requests two
  CTAs per cooperative group. TileIRAS chooses the SM100 layouts, partitions
  work across the CTA pair, creates the TMA/TMEM pipelines, inserts cluster
  synchronization, and emits the native instructions.

All implementations were compiled and disassembled. Both DSL kernels contain
native:

```text
UTMALDG.2D.2CTA
UTCHMMA.2CTA
UTMAREDG.2D.ADD
UCGABAR_ARV / UCGABAR_WAIT
```

CUDA S5 is the lower-shared-memory configuration used for the fused
production integration. CUDA S6 is the best isolated configuration and the
closest resource match to both DSL providers. CuTe DSL is the fastest
isolated implementation in the repeated campaign, while retaining explicit
control over the S6-like resource policy.

## 1. Scope

The report covers only isolated MLP3:

```text
dY: [T, H]
Z:  [T, I]
dA: [E, H, I]
```

Tokens are contiguous by expert. `expert_k_starts` and `expert_k_ends`
describe each expert's token interval in 64-token K-block units.

For expert `e`:

```text
dA[e, h, i] =
    sum(dY[token, h] * Z[token, i]
        for token in tokens_owned_by_expert_e)
```

The performance campaign uses:

```text
T = 8192
E = 8
BF16 inputs
FP32 MMA accumulation
BF16 reduction-add output
logical MMA tile = 256x256x64
k_split = 1
```

Correctness additionally covers split-K values 2 and 3, empty experts,
skewed experts, and non-power-of-two K-block counts.

## 2. What "2SM" means

A normal launch with two unrelated CUDA blocks is not a 2SM MMA.

The implementation requires:

1. Two CTAs in one cooperative group array (CGA).
2. Concurrent cluster scheduling of the CTA pair.
3. SM100 `cta_group::2` tensor-core instructions.
4. Pair-aware operand movement and synchronization.
5. A paired TMEM allocation visible to the hardware 2CTA MMA.

On B200, the relevant native instruction is:

```text
UTCHMMA.2CTA
```

Both CTAs contribute to one joined `256x256` result tile. The output ownership
is split so that each CTA handles 128 M rows. Within a CTA, epilogue
warpgroups process N chunks and reduce-add them to global memory.

The CTA pair must be launched as a cluster. An even grid alone would not
provide this contract.

## 3. CUDA/CuTe implementation

Sources:

```text
../../csrc/core/src/moe/mlp3_2sm.cuh
../bench_mlp3_2sm.cu
```

Exact snapshots of both sources used by this experiment are stored locally:

```text
backends/mlp3_2sm.cuh
backends/cuda_mlp3_2sm.cu
```

### 3.1 Compile-time topology

The CUDA path defines:

```cpp
using TileShape = Shape<Int<256>, Int<256>, Int<64>>;
using ClusterShape = Shape<_2, _1, _1>;
using AtomThrShape = Shape<_2, _1, _1>;

using TiledMma2Sm = decltype(make_tiled_mma(
    SM100_MMA_F16BF16_2x1SM_SS<...>{}));
```

Important consequences:

- `ClusterShape=(2,1,1)` forms one pair.
- `AtomThrShape=(2,1,1)` tells the pipeline that one MMA atom spans two CTAs.
- `SM100_MMA_F16BF16_2x1SM_SS` selects the SM100 two-CTA BF16/F16 UMMA atom.
- `CtaTileM=128`, so each CTA owns half of the joined M dimension.

### 3.2 Cluster launch

The host launcher uses `cudaLaunchKernelEx`:

```cpp
cudaLaunchAttribute cluster = {};
cluster.id = cudaLaunchAttributeClusterDimension;
cluster.val.clusterDim.x = 2;
cluster.val.clusterDim.y = 1;
cluster.val.clusterDim.z = 1;
```

The physical CUDA grid contains `2 * pairs` CTAs and is always even. Inside
the kernel:

```cpp
cta_rank = block_rank_in_cluster();
pair_rank = cta_rank % 2;
cell_start = blockIdx.x / 2;
cell_stride = gridDim.x / 2;
```

The two physical CTAs therefore share one logical persistent scheduler cell.

### 3.3 Pair-aware TMA mainloop

Each CTA obtains its hardware-defined slice:

```cpp
auto cta_mma = tiled_mma.get_slice(pair_rank);
```

The host constructs operands with:

```cpp
SM100_TMA_2SM_LOAD
make_tma_copy_A_sm100(...)
make_tma_copy_B_sm100(...)
```

The mainloop uses:

```cpp
PipelineTmaUmmaAsync<Stages, ClusterShape, AtomThrShape>
```

CUDA S5 has five SMEM stages; CUDA S6 has six. The producer warp issues
pair-aware TMA transfers into staged shared-memory layouts. Cluster-scoped
barriers prevent UMMA from consuming a stage before both CTA halves are
ready.

### 3.4 TMEM and 2CTA MMA

Both CTAs participate in:

```cpp
cute::TMEM::Allocator2Sm
```

The leader CTA's MMA warp issues the joined operation. The K=64 tile is
decomposed into four K=16 hardware MMA steps:

```cpp
for (int ks = 0; ks < size<2>(tCrDYT); ++ks) {
    tiled_mma.accumulate_ = first
        ? UMMA::ScaleOut::Zero
        : UMMA::ScaleOut::One;
    gemm(tiled_mma, a_fragment, b_fragment, accumulator);
}
```

`ScaleOut::Zero` initializes the TMEM accumulator on the first step;
`ScaleOut::One` accumulates all later K steps.

The explicit CUDA pipeline generates eight static `UTCHMMA.2CTA` sites in
the function because it materializes multiple control-flow regions. These are
static code sites, not eight MMAs for every K=64 iteration. A K=64 tile still
requires four dynamic K=16 MMA steps.

### 3.5 Accumulator and epilogue pipeline

The CUDA path uses two accumulator stages:

```cpp
PipelineUmmaAsync<2, AtomThrShape>
```

This allows the MMA warp to produce the next TMEM tile while epilogue
warpgroups read the current tile. Each CTA reads its 128-row TMEM half,
converts FP32 values to BF16, places output chunks in shared memory, and
issues:

```text
TMA_REDUCE_ADD
```

The native store instruction is:

```text
UTMAREDG.2D.ADD
```

This reduction contract supports multiple outer-split or split-K writers to
the same output tile.

## 4. cuTile implementation

Source:

```text
backends/cutile.py
```

The complete kernel is about a logical tile computation rather than a
thread/warp program.

### 4.1 Declaring a two-CTA CGA

The explicit 2SM request is:

```python
@ct.kernel(
    num_ctas=ct.ByTarget(sm_100=2),
    occupancy=ct.ByTarget(sm_100=1),
)
```

cuTile serializes this compiler option as:

```text
num_cta_in_cga = 2
```

The Python launch grid counts logical CGAs:

```python
pairs = min(sm_count // 2, cells)
ct.launch(stream, (pairs, 1, 1), kernel, args)
```

TileIRAS expands every logical block into two physical CTAs. This matches the
CUDA grid of `2 * pairs` physical CTAs.

The cuTile kernel never reads CTA rank because the compiler owns the
partitioning.

### 4.2 Persistent scheduler

The logical persistent scheduler is:

```python
block = ct.bid(0)
num_blocks = ct.num_blocks(0)

for cell_idx in range(block, total_cells, num_blocks):
    ...
```

One logical cell is:

```text
(expert, N tile, outer-split lane, split-K lane)
```

For a two-CTA CGA, `ct.bid()` is the logical pair index. This is equivalent to
the CUDA path's `blockIdx.x / 2`.

### 4.3 Operand loads

The source operations are:

```python
dy_tile = ct.load(
    dy,
    index=(kb, m_tile),
    shape=(64, 256),
    latency=10,
)

z_tile = ct.load(
    z,
    index=(kb, n_tile),
    shape=(64, 256),
    latency=10,
)
```

`latency=10` is a scheduling hint, not a request for a named instruction.
Given:

- an SM100 target;
- `num_ctas=2`;
- dense aligned tile access;
- the later 2CTA MMA consumer;

TileIRAS chooses pair-aware staged TMA and emits:

```text
UTMALDG.2D.2CTA
```

TileIRAS creates the shared-memory layouts, TMA descriptors, stage barriers,
producer role, and consumer waits internally.

### 4.4 MMA

The high-level source is:

```python
accumulator = ct.full((256, 256), 0.0, dtype=ct.float32)

accumulator = ct.mma(
    ct.transpose(dy_tile),
    z_tile,
    accumulator,
)
```

cuTile first represents this as a storage-free Tile IR operation:

```text
tile_mma
```

TileIRAS then performs:

1. SM100 layout selection.
2. Two-CTA partitioning.
3. Shared-memory operand staging.
4. TMEM placement for the accumulator.
5. K=64 decomposition into four K=16 instructions.
6. Accumulation predicate construction.

The result is one four-site loop body:

```text
UTCHMMA.2CTA
UTCHMMA.2CTA
UTCHMMA.2CTA
UTCHMMA.2CTA
```

The lower static site count than CUDA does not mean less mathematical work.
The CUDA compiler duplicates its four-site sequence across explicit
pipeline/control-flow regions; cuTile keeps one compiler-managed region.

### 4.5 Reduction-add output

The output view and store are:

```python
output_tiles = output.tiled_view((256, 256))

output_tiles.atomic_store_add(
    (expert * num_m_tiles + m_tile, n_tile),
    ct.astype(accumulator, output.dtype),
)
```

cuTile represents this as:

```text
tile_atomic_red_view
```

Because the update is a dense BF16 tile and the old value is unused,
TileIRAS lowers it to four TMA reduction-add stores:

```text
UTMAREDG.2D.ADD
```

The four sites cover the output chunks owned by the two CTA halves and their
epilogue partitions.

### 4.6 Compiler-inserted hardware mechanisms

There is no single source line for each item below:

| Generated mechanism | Derived from |
|---|---|
| CTA rank and operand ownership | `num_ctas=2`, tile shapes and target |
| CGA barriers | Cross-CTA dependencies of loads, MMA and TMEM |
| TMA descriptors | Array shape/stride plus tile load/store indices |
| Shared-memory layouts | MMA operand layouts and pipeline schedule |
| TMEM allocation | FP32 accumulator consumed by SM100 `tile_mma` |
| Mainloop stage count | Load latency hints, tile sizes and resource limits |
| Epilogue schedule | Atomic tiled output and accumulator lifetime |

This is the central abstraction difference: CUDA specifies these mechanisms;
cuTile specifies values and dependencies, then TileIRAS synthesizes the
mechanisms.

## 5. CuTe DSL implementation

Source:

```text
backends/cutedsl.py
```

CuTe DSL uses the same low-level design as CUDA/CuTe while moving the
implementation into a Python DSL.

### 5.1 Explicit two-CTA topology

The MMA atom is built with:

```python
self.cta_group = tcgen05.CtaGroup.TWO
tiled_mma = sm100_utils.make_trivial_tiled_mma(
    self.ab_dtype,
    self.a_major_mode,
    self.b_major_mode,
    self.acc_dtype,
    self.cta_group,
    self.mma_tiler[:2],
)
```

The kernel launches with:

```python
cluster=(2, 1, 1)
grid=(num_pairs * 2, 1, 1)
```

Unlike cuTile, CuTe DSL reads the physical block rank and explicitly selects
the CTA's MMA slice.

### 5.2 Explicit TMA and TMEM pipeline

The provider constructs SM100 pair-aware A/B TMA atoms from the cluster
shape and two-CTA MMA thread layout. It then creates:

```text
PipelineTmaUmma: 6 mainloop stages
PipelineUmmaAsync: 2 accumulator stages
TmemAllocator: two-CTA mode, 512 columns
```

The SMEM budget is:

```text
mainloop operands: 196,608 B
epilogue buffers:   32,768 B
pipeline budget:     1,024 B
total:             230,400 B
```

This is 768 B above CUDA S6 and 228 B below cuTile.

### 5.3 Persistent scheduler

The three implementations use the same logical cell:

```text
(expert, N tile, outer-split lane, split-K lane)
```

CuTe DSL launches `2 * num_pairs` physical CTAs and computes
`pair_idx = blockIdx.x // 2`. Each TMA, MMA, and epilogue role independently
walks:

```python
for cell_idx in range(pair_idx, total_cells, num_pairs):
    ...
```

The same `walk_begin`, `walk_end`, `kb_lo`, and `kb_hi` formulas are shared
with CUDA and cuTile.

### 5.4 Native reduction output

The epilogue creates:

```python
cpasync.CopyReduceBulkTensorTileS2GOp(cute.ReductionKind.ADD)
```

FP32 accumulators move from TMEM to registers, convert to BF16, stage through
SMEM, and issue a TMA reduction store. This preserves outer-split and split-K
semantics rather than replacing them with a non-overlapping store.

### 5.5 Generated code

CuTe DSL PTX contains:

```text
tcgen05.mma.cta_group::2
cp.async.bulk.tensor...cta_group::2
cp.reduce.async.bulk.tensor...add
```

ptxas lowers those operations to the same SASS classes used by CUDA and
cuTile: `UTCHMMA.2CTA`, `UTMALDG.*.2CTA`, `UTMAREDG.*.ADD`, and CGA
barriers.

## 6. Source-to-SASS correspondence

The finalized cuTile source maps as follows:

| Source | Tile IR meaning | Generated SM100 behavior |
|---|---|---|
| `backends/cutile.py:22-24` | Entry hints, `num_cta_in_cga=2` | Two-CTA CGA and cluster synchronization |
| `backends/cutile.py:36-37` | Logical block/grid IDs | Persistent CGA scheduler |
| `backends/cutile.py:43` | Dense `256x256` output view | Tiled epilogue address space |
| `backends/cutile.py:68-80` | Two dense tile loads | `UTMALDG.2D.2CTA` |
| `backends/cutile.py:82-86` | `tile_mma` | `UTCHMMA.2CTA` plus TMEM |
| `backends/cutile.py:88-91` | `tile_atomic_red_view` | `UTMAREDG.2D.ADD` |
| `backends/cutile.py:140-144` | Logical grid launch | `pairs` CGAs, two CTAs each |

The function-specific SASS counts are:

| Provider | 1CTA MMA | 2CTA MMA | 2CTA TMA load | TMA reduce-add | CGA barriers | Local memory |
|---|---:|---:|---:|---:|---:|---:|
| CUDA S5 | 0 | 8 | 4 | 4 | Present | 0 |
| CUDA S6 | 0 | 8 | 4 | 4 | Present | 0 |
| CuTe DSL | 0 | 4 | 4 | 4 | Present | 0 |
| cuTile | 0 | 4 | 4 | 4 | Present | 0 |

These assertions are generated by `inspect_codegen.py`. It:

1. Runs cuTile and CuTe DSL probes into clean compiler caches.
2. Extracts the actual CUBIN and Tile IR or PTX artifacts.
3. Runs `cuobjdump --dump-sass` on both DSL CUBINs.
4. Runs the same disassembler on both CUDA binaries.
5. Isolates only the provider's MLP3 2SM function.
6. Fails if 2CTA MMA, 2CTA TMA, CGA barriers or TMA reduction-add are absent.
7. Records SHA-256 hashes in `codegen.json`.

## 7. Apple-to-apple contract

The comparison matches the algorithm, topology, data type, work assignment
and timing protocol.

| Property | CUDA/CuTe | CuTe DSL | cuTile |
|---|---|---|---|
| Operation | `dY.T @ Z` per expert | Same | Same |
| Input layout | Row-major `[T,H]`, `[T,I]` | Same | Same |
| Input dtype | BF16 | BF16 | BF16 |
| Accumulation | FP32 | FP32 | FP32 |
| Output dtype | BF16 | BF16 | BF16 |
| Logical tile | `256x256x64` | `256x256x64` | `256x256x64` |
| Cooperative topology | Two-CTA cluster | `CtaGroup.TWO`, cluster 2 | `num_ctas=2` |
| Physical capacity | At most 74 pairs | Same | Same |
| Persistent scheduler | Grid-stride over logical pairs | Same | Same |
| Cell decomposition | Expert, N tile, outer split, split-K | Same | Same |
| Expert ranges | K-block start/end arrays | Same | Same |
| Performance split-K | 1 | 1 | 1 |
| Output semantics | TMA reduction add | TMA reduction add | Atomic tile add lowered to TMA reduction add |
| Split tuning | All divisors of M-tile count | Same | Same |
| Flop count | `2*T*H*I` | Same | Same |

### 7.1 Matching the logical grid

CUDA launches:

```text
physical_grid = 2 * pairs
cluster_dim = 2
logical_grid = physical_grid / 2
```

cuTile launches:

```text
logical_grid = pairs
num_ctas = 2
physical_grid = logical_grid * 2
```

CuTe DSL launches the same physical grid as CUDA and explicitly sets
`cluster=(2,1,1)`.

The resulting number of CTA pairs is identical.

### 7.2 Matching scheduler cells

With one pair per cluster, CUDA computes:

```text
walk_begin = split_lane * num_m_tiles / outer_split
walk_end   = (split_lane + 1) * num_m_tiles / outer_split
```

Both DSL kernels use the same formulas. Each provider therefore visits the
same expert/output/K cells for a selected outer split.

### 7.3 Matching inputs and outputs

All providers use:

```text
dY seed = 0x12345678
Z seed  = 0x9abcdef0
```

and the same deterministic integer hash-to-BF16 construction. Output is
zeroed before every launch because both implementations perform reduction
add rather than overwrite.

### 7.4 Matching cache and timing policy

Before every timed launch:

1. The BF16 output is zeroed.
2. A 256 MiB buffer is written, exceeding the B200 L2 size.
3. The start event is recorded.
4. Exactly one MLP3 kernel is launched.
5. The stop event is recorded.

Output clearing and L2 eviction are ordered before the start event and are
not included in latency.

Each provider receives:

```text
5 warmups per split
30 measured launches per split
5 provider-order-rotated rounds
```

CUDA events measure device execution, so Python versus C++ host dispatch
overhead is excluded.

### 7.5 What is intentionally not matched

Pipeline construction is the variable being evaluated:

| Property | CUDA S5 | CUDA S6 | CuTe DSL | cuTile |
|---|---:|---:|---:|---:|
| Mainloop stages | 5 | 6 | 6 | Compiler-selected |
| Accumulator stages | 2 | 2 | 2 | Compiler-selected |
| Shared memory | 196,736 B | 229,632 B | 230,400 B | 230,628 B |
| TMA/barrier schedule | Explicit | Explicit | Explicit | Synthesized |

This is why two CUDA controls are necessary:

- S5 represents the production-integrated lower-memory policy.
- S6 is the best isolated CUDA policy and has an almost identical resource
  footprint to both DSL providers.

The S6 comparison is the more strictly matched isolated-kernel comparison.

## 8. Correctness

The CUDA S5 and S6 binaries each passed all 29 checks in the existing
standalone suite.

CuTe DSL and cuTile each passed nine focused cases:

1. Balanced single-tile experts.
2. Balanced multi-tile output with `outer_split=2`.
3. Empty and skewed experts.
4. All tokens assigned to one expert.
5. Multiple empty experts with prime K-block counts.
6. Split-K 2 on skewed experts.
7. Split-K 3 on skewed experts.
8. Split-K 3 with all but one expert empty.
9. Split-K 3 with prime K-block counts.

For `k_split=1`, the worst mean elementwise relative error was 0.141%.
Split-K uses absmax normalization because independent BF16 partial sums
change reduction order.

One important cuTile compiler-facing correctness condition was found:

```python
if kb_hi <= kb_lo:
    continue
```

This explicit guard is required for over-partitioned split-K slices. A
runtime Tile IR loop must not be entered with an inverted K interval.

## 9. Performance methodology

| Item | Value |
|---|---|
| GPU | NVIDIA B200 |
| SM count | 148 |
| Driver | 580.105.08 |
| CUDA compiler | 12.9.86 |
| Target | `sm_100a` |
| CuTe DSL | 4.6.0 |
| cuTile | 1.5.0 |
| TileIRAS | 13.3.36 |
| Shapes | Six models |
| Tokens | 8192 |
| Experts | 8 |
| Samples | 30 per split per round |
| Warmups | 5 per split per round |
| Rounds | 5 |
| L2 eviction | 256 MiB before each launch |
| Aggregate | Median latency per provider, then six-model geomean |

Every provider independently sweeps all divisors of the M-tile count. This
avoids favoring one scheduler with another provider's split choice.

## 10. Performance

```text
speedup = baseline latency / candidate latency
```

| Model | CUDA S5 ms | CUDA S6 ms | CuTe DSL ms | cuTile ms | DSL / S6 | DSL / cuTile |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B | 0.034816 | 0.034816 | **0.030720** | 0.032768 | **1.1333x** | **1.0667x** |
| Qwen3-235B-A22B | 0.088064 | 0.084000 | **0.081952** | 0.090128 | **1.0250x** | **1.0998x** |
| Qwen3.5-122B-A10B | 0.053248 | 0.053248 | **0.049184** | 0.053280 | **1.0826x** | **1.0833x** |
| Llama-4-Scout-17B-16E | 0.477184 | **0.468992** | 0.474144 | 0.497664 | 0.9891x | **1.0496x** |
| Mixtral-8x7B | 0.668608 | **0.655424** | 0.667712 | 0.690224 | 0.9816x | **1.0337x** |
| Mixtral-8x22B | 1.118240 | 1.122336 | **1.108000** | 1.135632 | **1.0129x** | **1.0249x** |
| **Geomean** | - | - | - | - | **1.0361x** | **1.0593x** |

### 10.1 CuTe DSL against CUDA

CuTe DSL is:

- `1.0501x` CUDA S5;
- `1.0361x` CUDA S6;
- fastest on four of six shapes.

An independent repeat produced `1.0503x` and `1.0360x`, respectively. The
S6 comparison is especially meaningful because the pipeline depth,
accumulator depth, topology, output contract, and SMEM footprint are closely
matched.

### 10.2 CuTe DSL against cuTile

CuTe DSL is `1.0593x` cuTile in geomean and wins every shape. The repeat
campaign produced `1.0601x`.

Both compile to the same native instruction classes. The remaining gap is a
pipeline and scheduling result, not a failure by cuTile to use 2CTA hardware.
CuTe DSL makes its six mainloop stages, two accumulator stages, four
epilogue stages, two epilogue warpgroups, and 512-column TMEM allocation
explicit. TileIRAS owns the corresponding cuTile choices.

### 10.3 cuTile in the new interleaved campaign

The four-provider rerun supersedes the older three-provider table:

```text
cuTile / CUDA S5 = 0.9913x
cuTile / CUDA S6 = 0.9780x
```

The change from the earlier unlocked-clock run reinforces the report's
original caution: differences of a few percent require same-session,
rotating-order comparisons. The CuTe DSL advantage reproduced across two
complete campaigns.

## 11. Interpretation

### 11.1 What CuTe DSL demonstrates

CuTe DSL provides a middle-control implementation:

- Python source and JIT compilation;
- explicit 2CTA MMA topology;
- explicit TMA layouts and multicast masks;
- explicit six-stage mainloop;
- explicit TMEM allocation and accumulator pipeline;
- explicit TMA reduction output;
- PTX and SASS observability.

It slightly exceeds hand-controlled CUDA S6 in this isolated experiment
without giving up control of the hardware policy.

### 11.2 What cuTile abstracts successfully

From a short tile program, TileIRAS recovered:

- a real two-CTA cluster;
- native 2CTA UMMA;
- paired-CTA TMA operand loads;
- cluster synchronization;
- TMEM accumulation;
- TMA BF16 reduction-add stores;
- persistent scheduling;
- an S6-sized latency-hiding pipeline;
- no local-memory spills.

### 11.3 What cuTile does not expose

Stable cuTile does not directly expose:

- CTA rank or explicit per-CTA ownership;
- cluster barrier placement;
- TMA descriptor construction;
- SMEM layouts;
- exact mainloop stage count;
- explicit TMEM allocation;
- accumulator-stage count;
- a shared-memory budget.

The last point matters for integration. The compiler selected a 230.6 KiB
pipeline, which is excellent for isolated MLP3 but behaves like CUDA S6 from
a resource perspective.

In the fused backward kernel, the S6-sized footprint disables the
communication bounce buffer and regresses end-to-end performance. Therefore:

```text
isolated cuTile performance ~= CUDA S6
```

does not imply:

```text
fused cuTile performance > production CUDA S5
```

The missing abstraction is not 2SM execution; it is explicit control over
the resource/performance tradeoff required by a larger fused kernel.

### 11.4 Integration remains unresolved

CuTe DSL's 230.4 KiB budget, cuTile's 230.6 KiB budget, and CUDA S6's
229.6 KiB footprint all have the same fused-kernel risk: they can disable the
communication bounce buffer. The isolated CuTe DSL win does not establish an
end-to-end backward win. A five-stage CuTe DSL variant is the next relevant
integration experiment.

## 12. Reproduction

```bash
cd liger_cute_kernels/bench_opt/mlp3_cutile_2sm

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2

python benchmark.py
python inspect_codegen.py
```

Outputs:

| File | Contents |
|---|---|
| `backends/cutile.py` | Final stable-cuTile kernel |
| `backends/cutedsl.py` | Explicit two-CTA CuTe DSL kernel |
| `backends/mlp3_2sm.cuh` | Snapshot of the CUDA/CuTe 2SM implementation |
| `backends/cuda_mlp3_2sm.cu` | Snapshot of the standalone CUDA benchmark |
| `benchmark.py` | Correctness and five-round performance harness |
| `results.csv` | Compact performance table |
| `results_raw.json` | Every round, split winner and correctness result |
| `inspect_codegen.py` | CUBIN extraction and function-specific SASS checks |
| `codegen.json` | Instruction counts and artifact hashes |
| `artifacts/cutedsl.ptx` | CuTe DSL PTX |
| `artifacts/cutedsl.sass` | Disassembled CuTe DSL kernel |
| `artifacts/cutile.sass` | Disassembled cuTile kernel |
| `artifacts/cuda_s5.sass` | Disassembled CUDA S5 binary |
| `artifacts/cuda_s6.sass` | Disassembled CUDA S6 binary |

## 13. Conclusion

The experiment confirms that both Python DSLs generate a real Blackwell 2SM
MLP3 kernel rather than merely launching two independent CTAs.

The native-code and performance results support three conclusions:

1. CuTe DSL can match the explicit CUDA/CuTe design in Python and slightly
   exceed CUDA S6 in this isolated campaign.
2. cuTile abstracts the hardware execution mechanisms well: 2CTA UMMA, TMA,
   CGA synchronization, TMEM and reduction stores are all recovered from a
   high-level tile program.
3. cuTile does not yet expose integration-level resource policy as directly:
   its compiler-selected S6-like shared-memory footprint cannot be reduced to
   the production S5 budget through a direct stable API.

For isolated MLP3, CuTe DSL is the strongest implementation in this
comparison, while cuTile remains a compact native-code implementation. For
the fused communication-aware backward path, the lower-memory policy still
matters more than the isolated ranking.
