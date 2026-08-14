# Blackwell kernel stack

This standalone experiment contains the SM100 paired-CTA MLP1 kernel, its
correctness and throughput test, and the reports describing the broader
Blackwell LigerCute optimization stack.

## Layout

- `backends/`: experimental CUDA/CuTe kernel sources
- `docs/`: optimization strategy, results, and comparison reports
- `test_mlp1_fused_2sm.cu`: standalone correctness and throughput test

## Build

Set `CUTLASS_HOME` to a CUTLASS checkout, then run:

```bash
cmake -S . -B build
cmake --build build -j
./build/test_mlp1_fused_2sm
```
