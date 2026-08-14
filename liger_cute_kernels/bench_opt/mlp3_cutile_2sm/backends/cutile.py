"""Stable-cuTile implementation of the isolated MLP3 weight-gradient kernel.

The CUDA baseline computes, for each expert:

    dA[expert] = dY[expert].T @ Z[expert]

Both inputs are row-major BF16 tensors with tokens grouped contiguously by
expert. The output is BF16 and uses an atomic tile add, matching the baseline's
TMA_REDUCE_ADD output contract.
"""

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]

TILE_M = 256
TILE_N = 256
TILE_K = 64


@ct.kernel(
    num_ctas=ct.ByTarget(sm_100=2),
    occupancy=ct.ByTarget(sm_100=1),
)
def cutile_mlp3_2sm_kernel(
    dy,
    z,
    output,
    expert_k_starts,
    expert_k_ends,
    outer_split: ConstInt,
    k_split: ConstInt,
):
    """Compute persistent two-CTA MLP3 cells using compiler-managed pipelines."""
    block = ct.bid(0)
    num_blocks = ct.num_blocks(0)
    num_experts = expert_k_starts.shape[0]
    num_m_tiles = ct.cdiv(dy.shape[1], TILE_M)
    num_n_tiles = ct.cdiv(z.shape[1], TILE_N)
    total_cells = num_experts * num_n_tiles * outer_split * k_split

    output_tiles = output.tiled_view((TILE_M, TILE_N))
    zero_pad = ct.PaddingMode.ZERO

    for cell_idx in range(block, total_cells, num_blocks):
        k_slice = cell_idx % k_split
        cell_om = cell_idx // k_split
        chunk_idx = cell_om // outer_split
        split_lane = cell_om - chunk_idx * outer_split
        expert = chunk_idx // num_n_tiles
        n_tile = chunk_idx - expert * num_n_tiles

        walk_begin = split_lane * num_m_tiles // outer_split
        walk_end = (split_lane + 1) * num_m_tiles // outer_split
        kb_lo = ct.load(expert_k_starts, index=expert, shape=())
        kb_hi = ct.load(expert_k_ends, index=expert, shape=())
        k_total = kb_hi - kb_lo
        k_per_split = ct.cdiv(k_total, k_split)
        kb_lo = kb_lo + k_slice * k_per_split
        kb_hi = min(kb_lo + k_per_split, kb_hi)
        if kb_hi <= kb_lo:
            continue

        for m_tile in range(walk_begin, walk_end):
            accumulator = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
            for kb in range(kb_lo, kb_hi):
                dy_tile = ct.load(
                    dy,
                    index=(kb, m_tile),
                    shape=(TILE_K, TILE_M),
                    padding_mode=zero_pad,
                    latency=10,
                )
                z_tile = ct.load(
                    z,
                    index=(kb, n_tile),
                    shape=(TILE_K, TILE_N),
                    padding_mode=zero_pad,
                    latency=10,
                )
                accumulator = ct.mma(
                    ct.transpose(dy_tile),
                    z_tile,
                    accumulator,
                )

            output_tiles.atomic_store_add(
                (expert * num_m_tiles + m_tile, n_tile),
                ct.astype(accumulator, output.dtype),
            )


def _validate_inputs(dy, z, expert_k_starts, expert_k_ends):
    if dy.ndim != 2 or z.ndim != 2:
        raise ValueError("dy and z must be rank-2 tensors")
    if dy.shape[0] != z.shape[0]:
        raise ValueError("dy and z must have the same token dimension")
    if dy.dtype != torch.bfloat16 or z.dtype != torch.bfloat16:
        raise TypeError("dy and z must use torch.bfloat16")
    if dy.device != z.device or not dy.is_cuda:
        raise ValueError("dy and z must be CUDA tensors on the same device")
    if dy.shape[0] % TILE_K or dy.shape[1] % TILE_M or z.shape[1] % TILE_N:
        raise ValueError(f"shapes must be divisible by tile {(TILE_K, TILE_M, TILE_N)}")
    if expert_k_starts.shape != expert_k_ends.shape:
        raise ValueError("expert K-range tensors must have identical shapes")
    if expert_k_starts.dtype != torch.int32 or expert_k_ends.dtype != torch.int32:
        raise TypeError("expert K-range tensors must use torch.int32")
    if expert_k_starts.device != dy.device or expert_k_ends.device != dy.device:
        raise ValueError("expert K-range tensors must be on the input device")


def prepare(dy, z, expert_k_starts, expert_k_ends):
    _validate_inputs(dy, z, expert_k_starts, expert_k_ends)
    num_experts = expert_k_starts.numel()
    return {
        "dy": dy,
        "z": z,
        "expert_k_starts": expert_k_starts,
        "expert_k_ends": expert_k_ends,
        "output": torch.empty(
            (num_experts * dy.shape[1], z.shape[1]),
            dtype=torch.bfloat16,
            device=dy.device,
        ),
    }


def launch(state, outer_split, k_split=1):
    num_m_tiles = state["dy"].shape[1] // TILE_M
    if outer_split < 1 or num_m_tiles % outer_split:
        raise ValueError(f"outer_split={outer_split} must divide {num_m_tiles} M tiles")
    if k_split < 1:
        raise ValueError("k_split must be positive")
    num_experts = state["expert_k_starts"].numel()
    num_n_tiles = state["z"].shape[1] // TILE_N
    cells = num_experts * num_n_tiles * outer_split * k_split
    sm_count = torch.cuda.get_device_properties(state["dy"].device).multi_processor_count
    pairs = max(1, min(sm_count // 2, cells))
    ct.launch(
        torch.cuda.current_stream(state["dy"].device),
        (pairs, 1, 1),
        cutile_mlp3_2sm_kernel,
        (
            state["dy"],
            state["z"],
            state["output"],
            state["expert_k_starts"],
            state["expert_k_ends"],
            outer_split,
            k_split,
        ),
    )
    return state["output"]
