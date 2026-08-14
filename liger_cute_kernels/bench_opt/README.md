# Blackwell benchmark experiments

This directory contains standalone B200 (`sm_100a`) experiments and reports
for the Blackwell LigerCute kernel work on this branch. The experiments are
kept outside production `csrc/` and `tests/`: each subdirectory owns its
sources, build entry point, results, and documentation.

## Where to start

| Directory | Question answered | Start here |
|---|---|---|
| [`gate_up_swiglu/`](gate_up_swiglu/) | How do CUDA/CuTe, CuTe DSL, cuTile, and Triton compare for fused gate/up projection plus SwiGLU? | [`docs/gate_up_swiglu_comparison.md`](gate_up_swiglu/docs/gate_up_swiglu_comparison.md) |
| [`mlp3_cutile_2sm/`](mlp3_cutile_2sm/) | How do CUDA/CuTe, CuTe DSL, and cuTile express and perform the same isolated paired-CTA MLP3 weight-gradient operation? | [`2SM_BACKEND_COMPARISON.md`](mlp3_cutile_2sm/2SM_BACKEND_COMPARISON.md) |
| [`blackwell_kernel_stack/`](blackwell_kernel_stack/) | How was the broader LigerCute MoE stack ported and optimized for Blackwell, and what did each optimization contribute? | [`docs/blackwell_lck_optimization_report.md`](blackwell_kernel_stack/docs/blackwell_lck_optimization_report.md) |

These scopes are related but not interchangeable. `gate_up_swiglu` studies a
fused forward operation, `mlp3_cutile_2sm` isolates a backward weight-gradient
kernel, and `blackwell_kernel_stack` describes the production-oriented MoE
optimization sequence across forward and backward execution.

## Recommended reading order

1. Read the [Blackwell optimization strategy](blackwell_kernel_stack/docs/blackwell_lck_optimization_strategy.md)
   for the architectural context: warp specialization, TMEM accumulator
   buffering, backward epilogue pipelining, paired-CTA MLP3/MLP4, and mainloop
   tuning.
2. Read the [Blackwell port and optimization report](blackwell_kernel_stack/docs/blackwell_lck_optimization_report.md)
   for the measured effect of those changes on one and eight B200 GPUs.
3. Use the [detailed MoE comparison tables](blackwell_kernel_stack/docs/blackwell_moe_optimization_comparison.md)
   when per-shape or checkpoint-level numbers are needed.
4. Read the [gate/up report](gate_up_swiglu/docs/gate_up_swiglu_comparison.md)
   for the 1CTA frontend comparison and its timing-methodology caveats.
5. Read the [MLP3 2SM report](mlp3_cutile_2sm/2SM_BACKEND_COMPARISON.md)
   for the matched 2CTA topology, source-to-SASS evidence, and integration
   constraints. [`RESULTS.md`](mlp3_cutile_2sm/RESULTS.md) is its shorter
   measurement summary.

## Common directory layout

The experiments generally use:

- `backends/` for CUDA/CuTe, CuTe DSL, cuTile, and Triton implementations;
- `benchmark.py` for correctness and timed provider comparisons;
- `CMakeLists.txt` for standalone CUDA targets;
- `results.csv` for compact tables and `results_raw.json` for full protocol
  and per-round data;
- `codegen.json` and inspection scripts for generated-code evidence;
- `docs/` for current reports and, where present, `docs/archive/` for
  superseded investigations retained only as provenance.

Treat each report's stated protocol as part of its result. Do not compare
headline numbers across directories without accounting for different
operations, CTA topology, scheduler policy, cache treatment, split strategy,
and timing method.

## Reproduce

The experiments require an NVIDIA Blackwell GPU and their documented CUDA,
CUTLASS, CuTe DSL, cuTile, Triton, and Python dependencies.

```bash
# Fused gate/up + SwiGLU
cd liger_cute_kernels/bench_opt/gate_up_swiglu
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
python benchmark.py

# Isolated paired-CTA MLP3
cd ../mlp3_cutile_2sm
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
python benchmark.py
python inspect_codegen.py

# Standalone paired-CTA MLP1 from the Blackwell stack
cd ../blackwell_kernel_stack
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
./build/test_mlp1_fused_2sm
```
