# Blackwell LigerCute Optimization Strategy

## Architectural starting point

LigerCute implements a persistent fused expert-parallel MoE kernel. Dedicated
NVSHMEM warps move remote token tiles through a symmetric-memory staging ring
while the remaining warps execute the expert MLP. Forward fuses dispatch,
SwiGLU MLP1/MLP2, and combine; backward fuses gradient communication with
MLP1-act, MLP2-T, MLP3, MLP4, and MLP5.

The Blackwell port replaced Hopper WGMMA/register accumulators with SM100
UMMA/tcgen05 and TMEM accumulators while preserving the Hopper specialization.
The six principal operations are:

- MLP1: `U = X B`, `V = X C`, `Z = SiLU(U) * V`
- MLP2: `Y = Z A^T`
- MLP2-T: `dZ = dY A`
- MLP3: `dA = dY^T Z`
- MLP4: `dB = dU^T X`, `dC = dV^T X`
- MLP5: `dX = dU B + dV C`

## 1. Accumulator double buffering

The original SM100 path used one TMEM accumulator stage. Warp 4 issued UMMA
and also participated in the epilogue, so the next output tile could not begin
until the current TMEM result was drained. `PipelineUmmaAsync<2>` allocates two
TMEM banks. UMMA writes tile N+1 into one bank while epilogue warps drain tile N
from the other.

This is an output-side pipeline:

```text
time       t0             t1             t2
UMMA       N -> bank A    N+1 -> bank B  N+2 -> bank A
epilogue                  drain A        drain B
```

It is distinct from operand mainloop staging in shared memory.

## 2. Warp-role specialization

The controlled historical pair keeps `AccStages=2` constant.

Before (`448cbbe`):

```text
WG0: W0 TMA | W1 GET | W2 GET | W3 PUT
WG1: W4 UMMA-only | W5-W7 idle
WG2: W8-W11 epilogue
```

After (`a52b5f`):

```text
WG0: W0 TMA | W1 GET | W2 PUT | W3 UMMA-only
WG1: W4-W7 epilogue
WG2: W8-W11 epilogue
```

The optimization consolidates communication onto two warps, repurposes warp 3
as the epilogue-free UMMA producer, restores two full epilogue warpgroups, and
updates barrier participation. Therefore the measured effect is the complete
warp-role package, not “warp 3 alone.”

## 3. Backward epilogue pipelining

The same producer/consumer separation was extended across MLP2-T, MLP3, MLP4,
and MLP5. Warp 3 can issue the next UMMA K-loop while warps 4-11 drain the
previous TMEM stage in `EpiChunkN` pieces and overlap asynchronous TMA stores or
`TMA_REDUCE_ADD`. MLP4 computes dB and dC as sequential phases while reusing
its shared operand storage.

## 4. Paired-CTA 2SM MLP3/MLP4

The 2SM path is explicitly launched; CUDA does not infer it:

```text
cudaLaunchKernelEx
  clusterDim = (2, 1, 1)
        |
        +-- CTA rank 0 / SM 0
        +-- CTA rank 1 / SM 1
```

The grid must contain complete CTA pairs. Pair-aware SM100 TMA descriptors and
multicast masks populate each CTA's private shared-memory operand stages.
`Allocator2Sm` creates the paired TMEM allocation, and a `cta_group::2` UMMA
atom computes one joined logical tile. Each CTA drains its own 128-row output
half. Cluster synchronization surrounds allocation, execution, and release.

Cluster launch itself can add scheduling and barrier overhead. Benefits come
from enabling the paired UMMA tile, co-scheduling peers, pair-aware operand
movement, and coordinated TMEM—not from launch syntax alone.

## 5. Mainloop depth and tuning

Three depths must not be conflated:

| Knob | Storage | Purpose |
|---|---|---|
| `Stages3` | per-CTA SMEM | TMA operand prefetch ahead of UMMA |
| `AccStages` | TMEM | overlap UMMA output with epilogue |
| `CommNumStages` | symmetric HBM ring | overlap remote transport with compute |

Representative MLP3/MLP4 candidates use `TileK=64`, `Stages3=4/5`, and
`EpiChunkN34=64`. Offline tuning emits separate single- and multi-GPU SM100
tables; stage 5 is shape-selected rather than universal.

## Diagram briefs

### Warp specialization

Draw the two 12-warp layouts above side by side. Color TMA blue, NVSHMEM gray,
UMMA orange, and epilogue green. Under both, draw identical two-bank TMEM
timelines to show that accumulator depth is controlled.

### Accumulator double buffering

Draw TMEM banks A/B and three time columns. Alternate UMMA writes and epilogue
reads, labeling `producer_acquire`, `producer_commit`, `consumer_wait`, and
`consumer_release`.

### Paired-CTA 2SM

Draw a cluster boundary containing CTA0/SM0 and CTA1/SM1. Give each CTA private
SMEM slots S0-S4. Feed them through pair-aware TMA arrows, then join them at a
`cta_group::2 UMMA` box backed by paired TMEM. Split the result into two
per-CTA 128-row epilogues. Mark every cluster barrier.

### Five-stage mainloop

Draw a circular SMEM queue `[S0][S1][S2][S3][S4]`, with TMA filling a future K
tile while UMMA consumes the current one. Beneath it, draw only two TMEM banks.
Explicitly state that five SMEM stages do not mean five accumulators.

## Measured headline

| Comparison | Cases | Geomean | Median | Min | Max | Wins | Parity | Regressions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| single_A_to_W0_fwd | 35 | 0.977x | 0.973x | 0.861x | 1.214x | 11 | 2 | 22 |
| single_W0_to_W1_warp_specialization_fwd | 35 | 1.008x | 1.005x | 0.975x | 1.120x | 16 | 18 | 1 |
| single_A_to_W1_total_mlp1_fwd | 35 | 0.985x | 0.979x | 0.870x | 1.216x | 12 | 1 | 22 |
| single_W1_to_B_remaining_pipeline_fwd | 35 | 1.021x | 1.023x | 0.902x | 1.080x | 28 | 4 | 3 |
| single_A_to_B_fwd | 35 | 1.006x | 1.017x | 0.888x | 1.227x | 22 | 0 | 13 |
| single_A_to_B_bwd | 35 | 1.493x | 1.567x | 1.089x | 1.997x | 35 | 0 | 0 |
| single_A_to_B_fwd_plus_bwd | 35 | 1.401x | 1.443x | 1.081x | 1.845x | 35 | 0 | 0 |
| single_B_to_C_2sm_stage_tuning_fwd | 35 | 1.004x | 1.002x | 0.917x | 1.126x | 12 | 21 | 2 |
| single_B_to_C_2sm_stage_tuning_bwd | 35 | 1.062x | 1.044x | 0.997x | 1.183x | 33 | 2 | 0 |
| single_B_to_C_2sm_stage_tuning_fwd_plus_bwd | 35 | 1.049x | 1.035x | 0.997x | 1.140x | 33 | 2 | 0 |

For complete per-shape values, methodology, and limitations, see
`blackwell_moe_optimization_comparison.md`.
