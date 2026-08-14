"""Triton implementation of the isolated MLP3 weight-gradient kernel.

Computes, for each expert:

    dA[expert] = dY[expert].T @ Z[expert]

The performance path requests a two-CTA ``256x256x64`` tile. With
``k_split=1`` each program owns its output tile and can store it directly.
Split-K correctness specializations use BF16 atomic adds.
"""

import torch
import triton
import triton.language as tl

from triton.tools.tensor_descriptor import TensorDescriptor

TILE_M = 256
TILE_N = 256
TILE_K = 64

NUM_WARPS = 8
NUM_STAGES = 6
NUM_CTAS = 2


@triton.jit
def triton_mlp3_2sm_kernel(
    dy,
    z_desc,
    output_desc,
    output,
    expert_k_starts,
    expert_k_ends,
    hidden: tl.constexpr,
    intermediate: tl.constexpr,
    num_experts: tl.constexpr,
    num_m_tiles: tl.constexpr,
    num_n_tiles: tl.constexpr,
    k_split: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    pipeline_stages: tl.constexpr,
):
    """Compute one MLP3 output tile per logical two-CTA Triton program."""
    tile_index = tl.program_id(0)

    offsets_m = tl.arange(0, block_m)
    offsets_n = tl.arange(0, block_n)
    offsets_k = tl.arange(0, block_k)

    k_slice = tile_index % k_split
    output_tile = tile_index // k_split
    m_tile = output_tile % num_m_tiles
    expert_n_tile = output_tile // num_m_tiles
    n_tile = expert_n_tile % num_n_tiles
    expert = expert_n_tile // num_n_tiles

    kb_lo = tl.load(expert_k_starts + expert)
    kb_hi = tl.load(expert_k_ends + expert)
    k_total = kb_hi - kb_lo
    k_per_split = tl.cdiv(k_total, k_split)
    kb_lo += k_slice * k_per_split
    kb_hi = tl.minimum(kb_lo + k_per_split, kb_hi)

    if kb_hi > kb_lo:
        accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)
        for kb in tl.range(
            kb_lo,
            kb_hi,
            num_stages=pipeline_stages,
        ):
            token_offsets = kb * block_k + offsets_k
            m_offsets = m_tile * block_m + offsets_m
            n_offsets = n_tile * block_n + offsets_n
            dy_tile = tl.load(
                dy + m_offsets[:, None] + token_offsets[None, :] * hidden,
            )
            z_tile = z_desc.load((kb * block_k, n_tile * block_n))
            accumulator = tl.dot(
                dy_tile,
                z_tile,
                acc=accumulator,
            )

        output_offsets = (
            expert.to(tl.int64) * hidden * intermediate
            + (m_tile * block_m + offsets_m[:, None]).to(tl.int64) * intermediate
            + n_tile * block_n
            + offsets_n[None, :]
        )
        result = accumulator.to(output.dtype.element_ty)
        if k_split == 1:
            output_desc.store(
                (expert * hidden + m_tile * block_m, n_tile * block_n),
                result,
            )
        else:
            tl.atomic_add(
                output + output_offsets,
                result,
                sem="relaxed",
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
        raise TypeError("expert K-range tensors must use int32")
    if expert_k_starts.device != dy.device or expert_k_ends.device != dy.device:
        raise ValueError("expert K-range tensors must be on the input device")


def prepare(dy, z, expert_k_starts, expert_k_ends):
    _validate_inputs(dy, z, expert_k_starts, expert_k_ends)
    num_experts = expert_k_starts.numel()
    output = torch.empty(
        (num_experts, dy.shape[1], z.shape[1]),
        dtype=torch.bfloat16,
        device=dy.device,
    )
    return {
        "dy": dy,
        "z": z,
        "z_desc": TensorDescriptor.from_tensor(
            z,
            block_shape=(TILE_K, TILE_N),
        ),
        "output_desc": TensorDescriptor.from_tensor(
            output.view(num_experts * dy.shape[1], z.shape[1]),
            block_shape=(TILE_M, TILE_N),
        ),
        "expert_k_starts": expert_k_starts,
        "expert_k_ends": expert_k_ends,
        "output": output,
    }


def launch(state, outer_split, k_split=1):
    num_m_tiles = state["dy"].shape[1] // TILE_M
    if outer_split < 1 or num_m_tiles % outer_split:
        raise ValueError(f"outer_split={outer_split} must divide {num_m_tiles} M tiles")
    if k_split < 1:
        raise ValueError("k_split must be positive")

    num_experts = state["expert_k_starts"].numel()
    num_n_tiles = state["z"].shape[1] // TILE_N
    total_tiles = num_experts * num_m_tiles * num_n_tiles * k_split
    grid = (total_tiles,)
    triton_mlp3_2sm_kernel[grid](
        state["dy"],
        state["z_desc"],
        state["output_desc"],
        state["output"],
        state["expert_k_starts"],
        state["expert_k_ends"],
        hidden=state["dy"].shape[1],
        intermediate=state["z"].shape[1],
        num_experts=num_experts,
        num_m_tiles=num_m_tiles,
        num_n_tiles=num_n_tiles,
        k_split=k_split,
        block_m=TILE_M,
        block_n=TILE_N,
        block_k=TILE_K,
        pipeline_stages=NUM_STAGES,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        num_ctas=NUM_CTAS,
    )
    return state["output"]
