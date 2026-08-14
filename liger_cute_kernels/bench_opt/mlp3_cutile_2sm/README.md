# MLP3 CUDA, CuTe DSL, and cuTile 2SM experiment

This experiment compares four isolated Blackwell MLP3 weight-gradient
configurations:

```text
dA[expert] = dY[expert].T @ Z[expert]
```

All providers use BF16 inputs and output, a logical `256x256x64` MMA tile,
two CTAs per cooperative group, persistent grid-stride scheduling, identical
per-expert K-block ranges, independently tuned outer splits, and a reduction
add into the output tile.

The CuTe DSL and cuTile implementations also support the CUDA path's split-K
decomposition and validate repeated BF16 reduction-add writers. Performance
measurements hold `k_split=1`, matching the original standalone campaign.

## Providers

| Provider | 2SM expression | Pipeline control | Output |
|---|---|---|---|
| CUDA/CuTe S5 | Explicit `cta_group::2`, TMA, TMEM, barriers, S5/Acc2 | Programmer-controlled | `TMA_REDUCE_ADD` |
| CUDA/CuTe S6 | Same, with the standalone-optimal S6 pipeline | Programmer-controlled | `TMA_REDUCE_ADD` |
| CuTe DSL | Explicit `CtaGroup.TWO`, 6-stage TMA, 2-stage TMEM, barriers | Programmer-controlled Python DSL | TMA reduce-add |
| stable cuTile | `num_ctas=2`, `ct.mma`, load latency hints | Compiler-controlled | `TiledView.atomic_store_add` |

S5 is the lower-shared-memory production integration candidate. S6 is the
best isolated CUDA configuration and the closest resource match to both DSL
providers:

```text
CUDA S6:  229,632 B
CuTe DSL: 230,400 B
cuTile:    230,628 B
```

The exact CUDA/CuTe implementation and standalone driver used for the
measurements are copied into `backends/mlp3_2sm.cuh` and
`backends/cuda_mlp3_2sm.cu`. See
[`backends/CUDA_2SM_SNAPSHOT.md`](backends/CUDA_2SM_SNAPSHOT.md) for source
provenance and hashes.

CuTe DSL expresses CTA rank, cluster barriers, TMA, TMEM, and `tcgen05`
explicitly. cuTile intentionally contains none of those low-level mechanisms.
Generated-code inspection confirms that both paths produce native 2CTA MMA,
2CTA TMA loads, and TMA reduction stores.

## Outcome

See [`2SM_BACKEND_COMPARISON.md`](2SM_BACKEND_COMPARISON.md) for the detailed
CUDA/CuTe, CuTe DSL, and cuTile implementation, source-to-SASS mapping,
apples-to-apples methodology, and interpretation. [`RESULTS.md`](RESULTS.md)
is the shorter measurement-focused summary.

For the relationship between this 2CTA result, the gate/up 1CTA comparison,
and the broader Blackwell optimization reports, see the
[`bench_opt` navigation guide](../README.md).

In the repeated four-provider campaign, CuTe DSL is `1.0501x` CUDA S5,
`1.0361x` CUDA S6, and `1.0593x` cuTile. All providers emit the expected
native 2CTA instructions.

## Reproduce

Requirements are an SM100a GPU, CUDA/CuTe build dependencies, and
`cuda-tile[tileiras]>=1.5` plus `nvidia-cutlass-dsl>=4.6`.

```bash
cd liger_cute_kernels/bench_opt/mlp3_cutile_2sm
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
python benchmark.py
python inspect_codegen.py
```

The harness runs correctness first, then five rotating provider-order rounds
over six model shapes. Every split receives five warmups and 30 measured
launches. Before each launch, the output is zeroed and 256 MiB is written to
evict L2; CUDA events time only the MLP3 kernel.

Outputs:

- `results.csv`: compact performance comparison;
- `results_raw.json`: protocol, correctness, every split winner, and rounds.
- `codegen.json`: native-instruction assertions and artifact hashes.
