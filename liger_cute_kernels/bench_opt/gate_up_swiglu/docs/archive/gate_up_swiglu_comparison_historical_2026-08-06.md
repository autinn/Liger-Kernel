# Archived: fused gate/up + SwiGLU implementation study on B200

> **Status: superseded historical record.** This snapshot preserves the original
> experiments, including the old `1.205x` CUDA headline and recommendations that
> were written before all providers used identical input bits. Do not use those
> sections as the current cross-provider conclusion. See
> [`../gate_up_swiglu_comparison.md`](../gate_up_swiglu_comparison.md) for the
> corrected consolidated report.

`Z = SiLU(X @ W_gate^T) * (X @ W_up^T)` — the MoE MLP phase-1 ("gate/up projection +
SwiGLU"), *without* the down projection. This document collects every experiment run
while porting that kernel to the CUTLASS CuTe DSL and comparing it against the existing
Triton and hand-written CUDA C++ implementations.

**Headline question:** for a fused gate/up+SwiGLU kernel, which implementation strategy
wins, and how much of any gap is the *language* versus the *kernel design*?

## Consolidated rerun captured in this snapshot (2026-08-06)

The experiment now lives entirely under
`liger_cute_kernels/bench_opt/gate_up_swiglu/` and is reproduced with one
command: `python benchmark.py`. The two standalone activation providers are
pinned to Liger-Kernel PR #1271 (Triton) and PR #1277 (CuTeDSL).

The fused CUDA arm now uses the production 1CTA, `AccStages=2`, four-TMA-stage
pipeline with shape-specific N-splits. CuTeDSL uses 1CTA and explicit
`AccStages=2`; cuTile uses its closest 1CTA mode, with accumulator staging
chosen internally by TileIRAS.

| Model | CUDA fused | CuTeDSL fused | cuTile fused | Triton fused | cuBLAS + Triton | cuBLAS + CuTeDSL |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B | 1051.3 | 998.2 | 1030.8 | 698.7 | 865.0 | 999.4 |
| Qwen3-235B-A22B | 1027.8 | 976.8 | 1060.5 | 751.8 | 1116.7 | 1229.4 |
| Qwen3.5-122B-A10B | 1054.9 | 986.3 | 1070.0 | 728.3 | 1026.0 | 1148.0 |
| Llama-4-Scout-17B-16E | 1078.9 | 1120.7 | 1143.7 | 836.0 | 1298.4 | 1387.1 |
| Mixtral-8x7B | 1119.7 | 1082.8 | 1144.7 | 839.6 | 1330.3 | 1429.5 |
| Mixtral-8x22B | 1084.0 | 1076.5 | 1125.9 | 856.8 | 1339.4 | 1413.1 |
| **Geomean** | **1069.1** | **1038.7** | **1095.0** | **782.8** | **1148.3** | **1257.4** |
| **vs CUDA** | **1.000x** | **0.972x** | **1.024x** | **0.732x** | **1.074x** | **1.176x** |

### Conclusion at the time of this snapshot

1. **Production CUDA and explicit-AccStages CuTeDSL are within 3% geomean.**
   Their remaining difference includes CUDA's tuned N-split versus CuTeDSL's
   fixed persistent scheduler.
2. **cuTile's closest 1CTA mode is 2.4% above CUDA**, but TileIRAS does not
   expose an accumulator-stage control, so it cannot be called an exact
   `AccStages=2` match.
3. **Triton fused remains materially slower**. Its grouped 2-D grid materializes
   `pre_act[M,2I]` and does not reproduce the TMA/TMEM persistent pipeline used
   by the three Blackwell-native implementations.
4. **cuBLAS + PR #1277 CuTeDSL is fastest on this balanced workload**. Batched
   cuBLAS exploits equal `M/E` expert groups, while the vectorized CuTeDSL
   activation is cheaper than the PR #1271 Triton activation.
5. This is a **grouped/MoE comparison**, not a dense-SwiGLU ceiling. A dedicated
   `E=1` CUDA kernel could remove expert routing and retune its tile/pipeline.

The historical study below is retained because it documents how launch
overhead, scheduler mismatch, exact input bits, DVFS, accumulator staging and
epilogue warp allocation changed earlier headline numbers.

The exact historical raw data is archived as
`data/historical_fiveway_moe.json` and
`data/historical_fiveway_dense.json`.

> **Historical-study note:** §§0–6 preserve the original pre-warp-fix sweeps,
> including the best-known 1-CTA `0.85×` result. The authoritative post-fix
> five-provider rerun (fixed two-warpgroup CuTeDSL, natural 2-CTA) is in
> `Liger-Kernel/mlp1-5way-results.md`: **0.917× MoE / 1.009× dense**.

---

## 0. TL;DR

| # | Finding |
|---|---|
| 1 | The hand-written **CUDA C++ `mlp1_fused.cuh` is the fastest** implementation: **1.21× geomean** over the Triton+cuBLAS baseline. |
| 2 | **Triton's fused gate/up kernel is a large regression — 0.53×**, i.e. ~2× *slower* than simply not fusing. Fusion is not automatically a win. |
| 3 | Swapping only the activation to CuTeDSL (`cutedsl_swiglu_2gemm`) is a free **1.07×** — the GEMMs still dominate. |
| 4 | The **CuTeDSL fused port reaches 0.85×** of baseline at 1-CTA, i.e. it beats Triton-fused by **1.60×** but does not yet beat the unfused baseline or the C++ kernel. |
| 5 | Against the C++ kernel at *matched* tile/CTA/accumulator config, the DSL runs **0.84×** — so the irreducible language/codegen gap is ~16%, not the ~47% a naive comparison suggests. |
| 6 | **Pipelining (`AccStages=2`) is the single highest-value optimization** for the DSL: **1.11–1.13× mean (up to 1.33× per shape)**, and it is worth more to the DSL than to C++ (1.065×). |
| 7 | **2-CTA is not a free win.** It is ~neutral for the DSL at matched joined-M (1.00×) and *helps* it at its natural config (1.06–1.10×), but *hurts* both C++ kernels measured (MLP1 0.66–0.69×, MLP2 0.67–0.80×) — CTA-pairing must be co-designed with the tile scheduler. |

**Current best-known configuration: 1-CTA + pipelining (`AccStages=2`).** That is the
configuration used for the headline table in §2.

---

## 1. Method

| | |
|---|---|
| GPU | 1× NVIDIA B200 (`sm_100`, 183 GB), driver 580.105.08 |
| Toolchain | CUDA 13.0 (nvcc V13.0.88), CUTLASS 4.5.2, `nvidia-cutlass-dsl` 4.5.2 |
| Runtime | torch 2.11/2.13 +cu130, Triton 3.x |
| dtype | bfloat16 in/out, fp32 accumulate |
| Metric | `TFLOP/s = 4·M·H·I / seconds` — the two GEMMs only; SiLU FLOPs **not** counted |
| Timing | prime (JIT) → 10 warm-up → 50 CUDA-event-timed iters → **median**; 5 rounds |
| Shapes | `T = 8192` tokens, `E = 8` experts, per-model `(H, I)` |

Symbols: **T/M** = tokens (rows of `X`), **H** = hidden dim (the GEMM's K), **I** =
per-expert intermediate dim (the GEMM's N), **E** = experts.

Fairness controls applied (each was a real bug found and fixed — see §6):

- Both sides built on the **same** CUDA 13 + CUTLASS 4.5.2 stack (the shipped C++ binaries were CUDA 12).
- CuTeDSL **JIT is primed and synchronised before warm-up**, so `cute.compile` (590–740 ms) is never inside the timed region.
- Python-side launch cost (~40 µs) amortised via batching sized to a **~2 ms window** (a fixed batch count causes clock droop on long kernels).
- Per-CTA output tile **pinned to 128×128×64** for every cross-implementation cell.

---

## 2. HEADLINE — language / strategy comparison

**All five providers, one process, one harness, identical shapes.** CuTeDSL fused is at
its best-known config: **1-CTA + pipelined (`AccStages=2`)**.

### 2.1 Throughput (TFLOP/s) and speedup vs baseline

| model | H | I | Triton+cuBLAS (base) | | CuTeDSL+cuBLAS | | Triton fused | | **CUDA fused** | | CuTeDSL fused | |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| | | | TFLOP/s | × | TFLOP/s | × | TFLOP/s | × | TFLOP/s | × | TFLOP/s | × |
| Qwen3-30B-A3B | 2048 | 768 | 817.2 | 1.00× | 940.0 | 1.15× | 537.7 | 0.66× | **1154.6** | **1.41×** | 903.3 | 1.11× |
| Qwen3-235B-A22B | 4096 | 1536 | 1058.7 | 1.00× | 1120.9 | 1.06× | 589.2 | 0.56× | **1244.3** | **1.18×** | 942.6 | 0.89× |
| Qwen3.5-122B-A10B | 3072 | 1024 | 982.7 | 1.00× | 1056.8 | 1.08× | 580.8 | 0.59× | **1262.0** | **1.28×** | 934.2 | 0.95× |
| Llama-4-Scout-17B-16E | 5120 | 8192 | 1262.1 | 1.00× | 1327.2 | 1.05× | 599.0 | 0.47× | **1443.9** | **1.14×** | 941.5 | 0.75× |
| Mixtral-8x7B | 4096 | 14336 | 1261.3 | 1.00× | 1327.4 | 1.05× | 595.9 | 0.47× | **1480.3** | **1.17×** | 941.8 | 0.75× |
| Mixtral-8x22B | 6144 | 16384 | 1277.3 | 1.00× | 1345.8 | 1.05× | 606.7 | 0.47× | **1427.9** | **1.12×** | 930.5 | 0.73× |
| **geomean** | | | — | **1.00×** | — | **1.07×** | — | **0.53×** | — | **1.21×** | — | **0.85×** |

### 2.2 Ranking (geomean speedup vs baseline)

| rank | provider | strategy | geomean | vs baseline |
|---:|---|---|---:|---|
| 1 | `mlp1_pipelined` | **CUDA C++** fused, tcgen05 UMMA, `AccStages=2` | **1.21×** | +21% |
| 2 | `cutedsl_swiglu_2gemm` | 2 cuBLAS GEMMs + **CuTeDSL** activation | 1.07× | +7% |
| 3 | `triton_swiglu_2gemm` | 2 cuBLAS GEMMs + **Triton** activation | 1.00× | baseline |
| 4 | `mlp1_cutedsl_pipelined` | **CuTeDSL** fused, 1-CTA, `AccStages=2` | 0.85× | −15% |
| 5 | `triton_fused_gate_up` | **Triton** fused gate+up+act | 0.53× | −47% |

### 2.3 Wall-clock (ms, median)

| model | Triton+cuBLAS | CuTeDSL+cuBLAS | Triton fused | CUDA fused | CuTeDSL fused |
|---|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B | 0.0614 | 0.0530 | 0.0956 | **0.0446** | 0.0570 |
| Qwen3-235B-A22B | 0.1945 | 0.1815 | 0.3498 | **0.1655** | 0.2186 |
| Qwen3.5-122B-A10B | 0.1020 | 0.0960 | 0.1774 | **0.0816** | 0.1099 |
| Llama-4-Scout-17B-16E | 1.0686 | 1.0339 | 2.2800 | **0.9497** | 1.4130 |
| Mixtral-8x7B | 1.5230 | 1.4127 | 3.2032 | **1.2960** | 2.0378 |
| Mixtral-8x22B | 2.5395 | 2.4500 | 5.4315 | **2.3086** | 3.5112 |

### 2.4 Pairwise speedup matrix (geomean; read as ROW ÷ COLUMN)

| ↓row / col→ | Triton+cuBLAS | CuTeDSL+cuBLAS | Triton fused | CUDA fused | CuTeDSL fused |
|---|---:|---:|---:|---:|---:|
| **Triton+cuBLAS** | 1.00× | 0.93× | 1.87× | 0.82× | 1.18× |
| **CuTeDSL+cuBLAS** | 1.07× | 1.00× | 2.01× | 0.88× | 1.26× |
| **Triton fused** | 0.53× | 0.50× | 1.00× | 0.44× | 0.63× |
| **CUDA fused** | **1.21×** | **1.13×** | **2.28×** | 1.00× | **1.43×** |
| **CuTeDSL fused** | 0.85× | 0.79× | 1.60× | 0.70× | 1.00× |

### 2.5 Fraction of hardware peak (B200 dense bf16 ≈ 2250 TFLOP/s)

| model | Triton+cuBLAS | CuTeDSL+cuBLAS | Triton fused | CUDA fused | CuTeDSL fused |
|---|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B | 36% | 42% | 24% | **51%** | 40% |
| Qwen3-235B-A22B | 47% | 50% | 26% | **55%** | 42% |
| Qwen3.5-122B-A10B | 44% | 47% | 26% | **56%** | 42% |
| Llama-4-Scout-17B-16E | 56% | 59% | 27% | **64%** | 42% |
| Mixtral-8x7B | 56% | 59% | 26% | **66%** | 42% |
| Mixtral-8x22B | 57% | 60% | 27% | **63%** | 41% |

### 2.6 Diagnostics (not drop-in replacements)

| provider | geomean | note |
|---|---:|---|
| `triton_fused_no_preact` | 0.55× | Triton fused with the `pre_act[M,2I]` stores compiled out. Only +0.02× ⇒ **the extra output traffic is NOT why Triton-fused is slow**; the GEMM itself is. |
| `loop_swiglu_2gemm` | 0.82× | Per-expert Python loop of `2·E` `torch.mm` calls instead of 2 batched `bmm`. Launch-bound — this is why the headline baseline uses batched GEMMs (scoring against the loop would inflate every fused kernel by ~1.2–3×). |

### 2.7 Takeaways

1. **Fusion is not automatically a win.** Triton's fused kernel loses to *not fusing* by ~2×, and the `no_preact` diagnostic shows it is the grouped-GEMM inner loop, not the extra stores.
2. **The C++ kernel is the only implementation that beats the unfused baseline** — and by a solid 21%.
3. **The cheapest real win available today is provider #2**: keep cuBLAS for the GEMMs and swap only the activation to CuTeDSL for a free ~7%, with no kernel-design risk.
4. **The DSL fused port is not yet competitive with cuBLAS+activation**, but it is already 1.60× faster than Triton-fused and within 16% of C++ at matched config (§3).

---

## 3. Language deep-dive — DSL vs C++ at *matched* configuration

§2 compares implementations as they would actually be used. This section removes every
confound (tile shape, CTA mode, accumulator staging) so the residual is language/codegen only.

Per-CTA output tile pinned to **128×128×64** in all cells.

### 3.1 Matched 1-CTA comparison

| model | DSL acc=1 | CUDA acc=1 | ratio | DSL acc=2 | CUDA acc=2 | ratio |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B | 850.7 | 1060.0 | 0.803× | 1072.3 | 1156.2 | 0.927× |
| Qwen3-235B-A22B | 992.0 | 1201.8 | 0.825× | 1127.8 | 1277.1 | 0.883× |
| Qwen3.5-35B-A3B | 833.4 | 1010.6 | 0.825× | 847.2 | 1115.6 | 0.759× |
| Qwen3.5-122B-A10B | 946.7 | 1152.3 | 0.822× | 1084.3 | 1265.0 | 0.857× |
| Mixtral-8x7B | 1066.8 | 1362.1 | 0.783× | 1221.6 | 1415.8 | 0.863× |
| Mixtral-8x22B | 1146.4 | 1388.2 | 0.826× | 1074.5 | 1419.3 | 0.757× |
| Llama-4-Scout | 1057.3 | 1371.5 | 0.771× | 1187.5 | 1424.7 | 0.834× |
| **MEAN** | **984.8** | **1220.9** | **0.808×** | **1087.9** | **1296.2** | **0.840×** |

**The DSL codegen gap is ~16–19%.** The spread at `acc=1` is remarkably tight
(0.771–0.826×), which is the signature of a genuine codegen difference rather than a
tuning artifact.

### 3.2 Why the naive number was better than the honest one

| DSL configuration | vs CUDA |
|---|---:|
| best-of-sweep (2-CTA, tile 256×128) | 0.942× |
| **matched 1-CTA, tile 128×128** | **0.840×** |

~10 points of the DSL's apparent competitiveness came from it choosing **2-CTA UMMA** —
an optimization the C++ MLP1 does not implement — not from better generated code. Always
pin the tile before making a language claim.

### 3.3 Where the original (pre-optimization) gap came from

The first naive DSL port ran 0.41–0.90×. Decomposition:

| cause | evidence | verdict |
|---|---|---|
| **Host-side launch cost** (~40 µs/launch: 26.8 µs DLPack marshalling, 8.3 µs validation, 4.9 µs invoke; C++ ≈ 5 µs) | batching lifts the worst shape 0.407× → 0.799× | **measurement artifact**, removable via TVM-FFI fast path or CUDA graphs |
| **One output tile per CTA** (stock `dense_gemm.py` structure) | hold I=512, grow K: 0.407× (32 k-tiles) → 0.798× (128) → **0.981× (512)** | **real**, fixed by the persistent scheduler |
| MMA throughput | marginal cost 0.77 µs/k-tile (DSL) vs 0.76 (C++) | **not a factor** — parity within 1.3% |

Fixed-cost model fitted to the naive port: `t = 15.5 µs + 0.77 µs × k_tiles`, predicting
39% overhead at K=32 k-tiles and 3.8% at K=512 — exactly the measured shape sensitivity.
MoE shapes (H = 2048–6144 → 32–96 k-tiles) are the worst case for this bug.

---

## 4. Deep-dive: pipelining (`AccStages` = 1 vs 2)

`AccStages=2` double-buffers the TMEM accumulator so the MMA for tile *n+1* overlaps the
epilogue of tile *n*. Because a 1-SM accumulator is `TileN` columns wide and TMEM has only
512 columns, 2 stages × (U, V) requires **`4·TileN ≤ 512` ⇒ `TileN ≤ 128`** — the same
budget the C++ kernel encodes as `static_assert(AccStages * (2 * TileN) <= 512)`.

### 4.1 Gain from acc=1 → acc=2

| model | DSL 1-CTA | DSL 2-CTA | CUDA 1-CTA |
|---|---:|---:|---:|
| Qwen3-30B-A3B | 1.260× | 1.333× | 1.091× |
| Qwen3-235B-A22B | 1.137× | 1.105× | 1.063× |
| Qwen3.5-35B-A3B | 1.017× | 1.031× | 1.104× |
| Qwen3.5-122B-A10B | 1.145× | 1.160× | 1.098× |
| Mixtral-8x7B | 1.145× | 1.037× | 1.039× |
| Mixtral-8x22B | 0.937× | — | 1.022× |
| Llama-4-Scout | 1.123× | — | 1.039× |
| **MEAN** | **1.109×** | **1.133×** | **1.065×** |

### 4.2 Interpretation

- **Pipelining is the single highest-value optimization** for the DSL port: up to 1.33×.
- It is worth **more to the DSL (1.11–1.13×) than to C++ (1.065×)**, because the C++ kernel already loops over n-tiles inside each CTA — its per-tile prologue was amortised even at `AccStages=1`, so it gained *only* overlap, whereas the DSL gained amortisation *and* overlap.
- **Qwen3.5-35B-A3B (I=512) barely benefits (1.02×)**: with only 32 k-tiles there is not enough mainloop to hide the epilogue behind, no matter how it is staged. This is the one shape where the DSL stays at ~0.76–0.80×.
- The benefit **survives the tile match** (mean 1.122× at pinned 128×128), so it is not a tile-shape artifact.

---

## 5. Deep-dive: 1-CTA vs 2-CTA (a.k.a. 2-SM / `cta_group::2` / `2x1SM`)

### 5.1 What CTA mode actually changes

**2-CTA does *not* give each CTA a bigger tile.** Two CTAs in a cluster are welded together
to execute one MMA of twice the M extent, splitting M between them:

```
cta_tile_shape_m = mma_tiler_m / atom_thr_size      # ÷1 (1-CTA) or ÷2 (2-CTA)
```

| | MMA tile (one instruction) | CTA tile (per CTA) | TMEM/CTA (acc=2, U+V) |
|---|---|---|---|
| 1-CTA, tile 128×128 | 128×128 | 128×128 | 512 cols |
| 2-CTA, tile 256×128 | 256×128 | **128×128** | 512 cols |

Identical per-CTA budget. What 2-CTA buys: **the B operand (both weight matrices) is
TMA-multicast once to the CTA pair instead of fetched twice** — and in MoE MLP1 the weights
dominate memory traffic.

### 5.2 Effect of 1-CTA → 2-CTA

| model | DSL acc=1 | DSL acc=2 | CUDA acc=1 | CUDA acc=2 |
|---|---:|---:|---:|---:|
| Qwen3-30B-A3B | 0.997× | 1.055× | 0.685× | 0.644× |
| Qwen3-235B-A22B | 0.971× | 0.944× | 0.693× | 0.657× |
| Qwen3.5-35B-A3B | 0.990× | 1.005× | 0.753× | 0.656× |
| Qwen3.5-122B-A10B | 1.091× | 1.105× | 0.696× | 0.672× |
| Mixtral-8x7B | 0.966× | 0.874× | 0.628× | — |
| **MEAN** | **1.003×** | **0.997×** | **0.691×** | **0.657×** |

*(DSL 2-CTA here is at joined M=128 to match the CUDA 2SM binary exactly. At joined M=256 —
its natural config — the DSL gains 1.055× (acc=1) / 1.104× (acc=2) instead.)*

### 5.3 The C++ 2-SM regression, corroborated

A 2-SM MLP1 consumer was written from scratch for this comparison
(`csrc/core/src/moe/mlp1_fused_2sm.cuh`, correctness mean_rel 0.10–0.11%). It is
**33–37% slower than the 1-SM path**. This is not a one-off:

| kernel | 1-SM | 2-SM | ratio |
|---|---:|---:|---:|
| MLP2 (pre-existing, both arms from one source, numerically identical) | 674–802 | 460–538 | **0.67–0.80×** |
| MLP1 (written for this study) | 1060–1425 | 727–856 | **0.63–0.69×** |

**Root cause is scheduling, not the MMA.** The 1-SM path's N-split sweep finds its optimum
at **128 CTAs / 148 SMs**, but 2-SM forces a 2-CTA cluster grid (`grid.x = num_m_tiles*2`),
whose best point used 1024 CTAs. The forced cluster fights the existing occupancy tuning.

### 5.4 Interpretation

- **2-CTA helps the DSL and hurts C++.** The DSL's persistent scheduler and multicast were designed around clusters from the start; both C++ kernels bolt clusters onto a scheduler tuned for 1-SM.
- **Conclusion: stay on 1-CTA for now.** 2-CTA is only worth revisiting alongside a scheduler redesign.
- Practical asymmetry worth noting: trying 2-CTA in the DSL is a one-line parameter change; in C++ it was a ~250-line new consumer with documented deadlock/IMA traps.

---

## 6. Methodology bugs found (each changed a headline number)

| # | Bug | Impact | Fix |
|---|---|---|---|
| 1 | Prebuilt C++ binaries were **CUDA 12**, DSL JITs through **CUDA 13** | unfair compiler comparison | rebuilt C++ from source on CUDA 13 + CUTLASS 4.5.2 |
| 2 | CuTeDSL **JIT (590–740 ms)** could land in the timed region | catastrophic if unprimed | prime + `synchronize()` **before** warm-up |
| 3 | **~40 µs/launch Python host cost** inside CUDA-event window | worst shape read 0.407× instead of 0.799× | batch launches; report device-only separately |
| 4 | Fixed batch count caused **clock droop** on multi-ms kernels | −15% (1228 → 1050 TFLOP/s; 50 ms cooldown restored 1233) | size batch to a ~2 ms window |
| 5 | **Expert-routing granularity**: block is `mma_tiler_m` (256 under 2-CTA), not 128 | silent wrong-expert routing | `expert_block_size()` shared by kernel, harness and reference |
| 6 | DSL swept configs while C++ was pinned | overstated DSL by ~10 points | matched-tile table (§3.1) as the language claim |
| 7 | 2-SM epilogue reused **stale `partition_S` views** across accumulator stages | mean_rel 52300% | rebuild the partition per stage (`.data()` on a built view is a no-op) |

**Warm-up was ruled out as a confound**: 10 → 3000 warm-ups changes results by ~1%
(0.0778 → 0.0770 ms), far too little to explain any gap. Both harnesses use the identical
protocol: 1 prime + 10 warm-up + 50 event-timed iterations + median, sweeping launch configs
and reporting the peak.

---

## 7. Artifacts

| artifact | location |
|---|---|
| CuTeDSL kernel (naive + persistent/pipelined) | `Liger-Kernel/src/liger_kernel/ops/cutedsl/ops/fused_swiglu_gate_up.py` |
| DSL-vs-C++ harness | `Liger-Kernel/benchmark/scripts/benchmark_swiglu_gate_up_cutedsl.py` |
| 5-provider harness | `Liger-Kernel/benchmark/scripts/swiglu_pipeline/bench_mlp1_5way.py` (`--cutedsl-cta {1,2}`, `--cutedsl-acc-stages {1,2}`, `--cutedsl-epi-warpgroups {1,2}`) |
| 5-provider results | `Liger-Kernel/mlp1-5way-results.md` |
| C++ 1-SM kernel (reference) | `liger_cute_kernels/csrc/core/src/moe/mlp1_fused.cuh` |
| **C++ 2-SM kernel (new)** | `liger_cute_kernels/csrc/core/src/moe/mlp1_fused_2sm.cuh` |

Correctness of the CuTeDSL kernel: **mean_rel ≤ 0.003%** vs an fp32 torch reference across
all shapes, both CTA modes, both `AccStages`, with random MoE expert routing
(harness-reported mean rel err 8.0e-06 / 3.7e-06).

### Reproduce

```bash
export PYTHONPATH=/path/to/Liger-Kernel/src
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

# Current post-fix five-provider run
cd Liger-Kernel/benchmark/scripts/swiglu_pipeline
python bench_mlp1_5way.py --shapes moe --repeats 5 \
    --cutedsl-cta 2 --cutedsl-acc-stages 2 --cutedsl-epi-warpgroups 2 \
    --mlp1-bin <path>/test_mlp1_fused

# Historical §2 configuration: --cutedsl-cta 1 --cutedsl-epi-warpgroups 1

# §3 DSL vs C++ head-to-head
cd Liger-Kernel
python benchmark/scripts/benchmark_swiglu_gate_up_cutedsl.py --baseline-bin <path>/test_mlp1_fused
```

Building the C++ arms (out-of-tree):

```bash
cd liger_cute_kernels
export CUTLASS_HOME=/path/to/cutlass-4.5.2 CUDACXX=/usr/local/cuda-13.0/bin/nvcc
cmake -S . -B /tmp/build -G Ninja -DLIGER_CUTE_TESTS_ONLY=ON -DLIGER_CUTE_BUILD_TESTS=ON \
      -DLIGER_CUTE_CUDA_ARCH=100a -DGTest_DIR=/path/to/gtest/lib64/cmake/GTest
cmake --build /tmp/build --target test_mlp1_fused -j
# AccStages=1 control arm:  -DCMAKE_CUDA_FLAGS="-DMLP1_ACC_STAGES=1"
```

---

## 8. Recommendations

1. **Ship the C++ `mlp1_fused.cuh`** for fused gate/up+SwiGLU — 1.21× over baseline, the only implementation that beats not fusing.
2. **Adopt `cutedsl_swiglu_2gemm` opportunistically** — a free 1.07× for a one-line activation swap, no kernel-design risk.
3. **Do not ship `triton_fused_gate_up`** in its current form for these shapes; it is ~2× slower than not fusing, and the `no_preact` diagnostic shows the output contract is not the cause.
4. **Keep 1-CTA.** Revisit 2-CTA only together with a cluster-aware tile scheduler.
5. **Always enable `AccStages=2`** (requires `TileN ≤ 128`) — up to 1.33×, the best return per line of code.
6. **To close the remaining DSL gap**, in priority order: (a) TVM-FFI fast path to remove the ~40 µs launch cost, (b) investigate the I≤512 / low-k-tile regime where pipelining cannot help, (c) accept the ~16% codegen gap as the current cost of the DSL.
