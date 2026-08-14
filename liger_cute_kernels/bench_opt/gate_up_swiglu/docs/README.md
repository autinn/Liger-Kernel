# Gate/up SwiGLU documentation

This directory documents the B200 comparison for:

```text
gate = X @ W_gate[expert].T
up   = X @ W_up[expert].T
Z    = SiLU(gate) * up
```

## Current documents

| Document | Purpose |
|---|---|
| [`gate_up_swiglu_comparison.md`](gate_up_swiglu_comparison.md) | Authoritative results, protocol, caveats, and explanation of the superseded CUDA headline |
| [`gate_up_fused_backend_comparison.md`](gate_up_fused_backend_comparison.md) | Focused CUDA, CuTeDSL, cuTile, and Triton fused-provider performance and PTX/SASS evidence |
| [`cutile_vs_triton_complexity_assessment.md`](cutile_vs_triton_complexity_assessment.md) | cuTile/Triton comparison across Liger-Kernel PRs #1250, #1269, and #1321 |
| [`../../BLACKWELL_KERNEL_STACK_COMPARISON.md`](../../BLACKWELL_KERNEL_STACK_COMPARISON.md) | Cross-experiment synthesis of CUDA/CuTe, CuTe DSL, cuTile, and Triton using gate/up 1CTA and MLP3 2CTA evidence |

Start with the authoritative gate/up report for this experiment. Use the
top-level Blackwell stack comparison for cross-experiment conclusions; the
remaining documents answer narrower implementation and cross-kernel questions.

## Reproduce

```bash
cd liger_cute_kernels/bench_opt/gate_up_swiglu
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
python benchmark.py
```

Current outputs are stored one level above this directory:

| Artifact | Contents |
|---|---|
| `results.csv` | Compact six-provider performance table |
| `results_raw.json` | Protocol, correctness, commands, and per-round timings |
| `codegen.json` | Generated-code assertions and artifact hashes |

## Archive

`archive/` contains superseded experiments and forensic history. These files
are retained for provenance, not as current recommendations:

| Document | Historical scope |
|---|---|
| [`archive/matched_persistent_2cta.md`](archive/matched_persistent_2cta.md) | Controlled persistent-2CTA frontend comparison |
| [`archive/cutedsl_vs_cuda_forensic.md`](archive/cutedsl_vs_cuda_forensic.md) | Long-form PTX/SASS, scheduler, DVFS, and methodology investigation |
| [`archive/gate_up_swiglu_comparison_historical_2026-08-06.md`](archive/gate_up_swiglu_comparison_historical_2026-08-06.md) | Original superseded implementation study |

Historical CSV/JSON evidence is under `archive/data/`.
