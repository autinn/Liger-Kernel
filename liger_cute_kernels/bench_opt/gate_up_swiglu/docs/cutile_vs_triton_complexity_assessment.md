# cuTile versus Triton across kernel complexity

Sources: Liger-Kernel PRs
[#1250](https://github.com/linkedin/Liger-Kernel/pull/1250),
[#1269](https://github.com/linkedin/Liger-Kernel/pull/1269),
[#1321](https://github.com/linkedin/Liger-Kernel/pull/1321), and this fused
grouped SwiGLU experiment. All measurements are NVIDIA B200 BF16.

## Speedup assessment

```text
speedup = Triton latency / cuTile latency
```

Values above `1.0x` favor cuTile. PR geomeans cover four problem sizes.

| Operation | Structure | Source | cuTile / Triton | Per-shape range |
|---|---|---|---:|---:|
| GEGLU | Elementwise gate; no GEMM | #1250 | **1.009x** | 0.988-1.020x |
| Llama-4 RoPE | Elementwise gather/rotate | #1269 | **0.936x** | 0.667-1.113x |
| Qwen2-VL M-RoPE | Gather-heavy elementwise | #1269 | **1.069x** | 0.988-1.132x |
| RoPE | Elementwise gather/rotate | #1269 | **1.141x** | 1.082-1.226x |
| KL divergence | Single-pass reduction | #1269 | **1.537x** | 1.508-1.554x |
| Layer norm | Reduction + affine | #1250 | **1.227x** | 1.087-1.290x |
| Group norm | 2-D reduction + affine | #1269 | **1.377x** | 1.055-1.563x |
| Sparsemax | Sort + cumulative-sum reduction | #1269 | **1.107x** | 0.960-1.204x |
| Multi-token attention | Mask/softmax/conv/mask pipeline | #1269 | **1.166x** | 1.159-1.174x |
| Cross entropy | Feature-rich fused reduction | #1250 | **1.017x** | 1.002-1.037x |
| Fused linear JSD | Chunked GEMM + JSD reduction | #1250 | **1.734x** | 1.578-1.866x |
| DyT | Elementwise tanh + affine | #1321 | **1.501x** | 1.392-1.559x |
| Fused add + RMSNorm | Residual add + normalization | #1321 | **1.244x** | 1.001-1.343x |
| Fused linear cross entropy | PyTorch GEMMs + cuTile CE | #1321 | **2.924x** | 2.317-3.394x |
| GRPO, token IS | Policy-loss reduction | #1321 | **1.000x** | 1.000-1.000x |
| GRPO, sequence IS | Policy-loss reduction | #1321 | n/a | No Triton rows |
| PolyNorm | Multi-power normalization | #1321 | **1.269x** | 1.189-1.306x |
| RMSNorm | Reduction + affine | #1321 | **0.953x** | 0.455-1.450x |
| Softmax | Online reduction | #1321 | **1.206x** | 0.984-1.318x |
| Standalone SwiGLU | Elementwise SiLU-multiply; no GEMM | #1321 | **0.975x** | 0.903-1.005x |
| Fused grouped SwiGLU | Two grouped GEMMs + activation | This experiment | **1.399x** | 1.314-1.475x |

PR rows measure full forward+backward. The fused grouped row is forward-only,
so compare the ratios qualitatively rather than pooling them statistically.

## Standalone versus fused SwiGLU

| Operation | cuTile / Triton | Main distinction |
|---|---:|---|
| Standalone SwiGLU | **0.975x** | Elementwise activation only |
| Fused grouped gate/up SwiGLU | **1.399x** | Two expert GEMMs, persistent scheduling, TMA/TMEM pipeline |

This supports a narrow conclusion: cuTile is near Triton for standalone SwiGLU
but clearly faster for this fused grouped implementation. It does **not** prove
that cuTile advantage grows monotonically with complexity.

Counterexamples:

- Complex cross entropy is near parity (`1.017x`).
- Simple KL divergence favors cuTile by `1.537x`.
- RMSNorm ranges from `0.455x` to `1.450x` by shape.

## PR #1321 versus PyTorch/HuggingFace

PR #1321 is open at commit
`8f7b06d49b815726314372ca91129e13d6f03bfa`.

```text
speedup = PyTorch/HuggingFace latency / cuTile latency
```

Memory is full-operation peak at the largest reported shape.

| Operation | cuTile / PyTorch-HF | cuTile MB | Triton MB | PyTorch-HF MB |
|---|---:|---:|---:|---:|
| DyT | **1.917x** | 324.7 | 324.7 | 960.1 |
| Fused add + RMSNorm | **3.083x** | 578.4 | 578.4 | 1344.1 |
| Fused linear cross entropy | **0.221x** | 4328.1 | 6330.7 | 8208.1 |
| GRPO, token IS | **0.293x** | 7350.1 | 7350.1 | 19294.1 |
| GRPO, sequence IS | **0.293x** | 7350.1 | n/a | 19294.1 |
| PolyNorm | **10.042x** | 320.1 | 256.1 | 2048.1 |
| RMSNorm | **2.646x** | 258.4 | 258.4 | 1216.1 |
| Softmax | **0.920x** | 320.0 | 320.0 | 384.0 |
| Standalone SwiGLU | **1.026x** | 2000.0 | 2000.0 | 2560.0 |

Notable tradeoffs:

- Fused linear CE is `2.924x` Triton but about 4.5 times slower than eager
  PyTorch; it uses 47% less peak memory than PyTorch.
- GRPO matches Triton but is about 3.4 times slower than PyTorch; it uses 62%
  less peak memory.
- Softmax beats Triton in geomean but trails eager PyTorch.

## Scope

None of PRs #1250, #1269, or #1321 contains a plain cuTile GEMM or grouped GEMM
benchmark. PR #1321's fused-linear CE uses PyTorch matrix operations:

```python
logits = input_chunk @ weight.t()
grad_input = grad_logits @ weight
grad_weight += torch.mm(grad_logits.t(), input_chunk)
```

cuTile implements its cross-entropy portion, not the GEMMs.

The fused grouped SwiGLU providers also differ structurally:

| cuTile | Triton |
|---|---|
| Fixed 148-block persistent scheduler | One program per `(M,N)` tile |
| TileIRAS-managed TMA pipeline | `tl.dot` with autotuned input stages |
| Writes only `Z` | Writes `pre_act` and `Z` |
| Compiler-managed accumulator placement | No explicit TMEM double-buffer control |

Removing Triton's `pre_act` output historically helped only slightly. The
remaining gap primarily reflects scheduling, locality, and the Blackwell
pipeline design—not an isolated language/compiler effect.

## Caveats

- PR #1321 is open and can change.
- PR benchmark rows are full forward+backward; fused grouped SwiGLU is
  forward-only.
- The PR timing harness does not perform numerical assertions itself.
- Some small-shape results are launch-overhead dominated.
- The fused providers have different output and scheduling contracts.

## Sources

| Source | Evidence |
|---|---|
| [PR #1250](https://github.com/linkedin/Liger-Kernel/pull/1250) | GEGLU, layer norm, cross entropy, fused linear JSD |
| [PR #1269](https://github.com/linkedin/Liger-Kernel/pull/1269) | RoPE family, KL divergence, group norm, sparsemax, attention |
| [PR #1321](https://github.com/linkedin/Liger-Kernel/pull/1321) | DyT, normalization, fused linear CE, GRPO, softmax, standalone SwiGLU |
| `Liger-Kernel/benchmark/data/all_benchmark_data_cutile.csv` | PR speed and memory rows |
| [`gate_up_fused_backend_comparison.md`](gate_up_fused_backend_comparison.md) | Fused grouped performance and implementation |
| [`../../BLACKWELL_KERNEL_STACK_COMPARISON.md`](../../BLACKWELL_KERNEL_STACK_COMPARISON.md) | Cross-experiment programming-model synthesis |
| [`../results.csv`](../results.csv) | Fused grouped raw performance table |
