# CUDA MLP3 2SM source snapshot

This directory contains the exact CUDA/CuTe sources used as the comparison
baseline:

| File | Original source | SHA-256 |
|---|---|---|
| `cuda_mlp3_2sm.cu` | `../../bench_mlp3_2sm.cu` | `dc5e8705a3e65a214dc946e67fe9c6963379f2a37c9850d98b5320850e408597` |
| `mlp3_2sm.cuh` | `../../../csrc/core/src/moe/mlp3_2sm.cuh` | `19faf2accf22df1bff9b252cb5914166500bc486d3dd8086b5c4ae40d9818093` |

`CMakeLists.txt` builds these local copies for both S5 and S6, so the
experiment preserves the CUDA implementation even if the production source
later changes. The copied header still reuses the shared one-SM MLP3 helpers
from `csrc/core/src/moe/mlp3.cuh`.
