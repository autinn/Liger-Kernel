"""Triton grouped-GEMM fused gate/up SwiGLU provider.

Adapted from Liger-Kernel's SonicMoE ``_fused_up_proj_swiglu_kernel``:
https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_moe_kernels.py

Only the host routing setup is simplified because this comparison already uses
balanced, expert-blocked token rows.
"""

import torch
import triton
import triton.language as tl

BLOCK_M = 128


def _gemm_configs():
    return [
        triton.Config(
            {"BLOCK_N": block_n, "BLOCK_K": block_k},
            num_warps=num_warps,
            num_stages=num_stages,
        )
        for block_n in (64, 128)
        for block_k in (32, 64)
        for num_warps in (4, 8)
        for num_stages in (2, 3, 4, 5)
    ]


@triton.autotune(
    configs=_gemm_configs(),
    key=["H_dim", "I_dim", "BLOCK_M"],
)
@triton.jit
def fused_gate_up_swiglu_kernel(
    x_ptr,
    gate_up_ptr,
    x_gather_idx_ptr,
    expert_start_ptr,
    tile_row_start_ptr,
    tile_expert_ptr,
    pre_act_ptr,
    output_ptr,
    H_dim: tl.constexpr,
    I_dim: tl.constexpr,
    stride_x_T,
    stride_x_H: tl.constexpr,
    stride_w_E,
    stride_w_N,
    stride_w_K: tl.constexpr,
    stride_pre_TK,
    stride_pre_N: tl.constexpr,
    stride_output_TK,
    stride_output_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Copy of Liger's SonicMoE fused gate/up forward kernel."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    row_start = tl.load(tile_row_start_ptr + pid_m)
    expert_idx = tl.load(tile_expert_ptr + pid_m).to(tl.int64)
    n_start = pid_n * BLOCK_N
    expert_end = tl.load(expert_start_ptr + expert_idx + 1)

    m_offs = tl.arange(0, BLOCK_M)
    n_offs = tl.arange(0, BLOCK_N)
    k_offs = tl.arange(0, BLOCK_K)
    row_offs = (row_start + m_offs).to(tl.int64)
    row_mask = row_offs < expert_end

    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    n_idx = n_start + n_offs
    n_mask = n_idx < I_dim
    token_idx = tl.load(
        x_gather_idx_ptr + row_offs,
        mask=row_mask,
        other=0,
    ).to(tl.int64)

    for k in tl.range(0, H_dim, BLOCK_K):
        k_idx = k + k_offs
        k_mask = k_idx < H_dim
        x_ptrs = x_ptr + token_idx[:, None] * stride_x_T + k_idx[None, :] * stride_x_H
        x_tile = tl.load(
            x_ptrs,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )

        weight_mask = n_mask[:, None] & k_mask[None, :]
        gate_ptrs = gate_up_ptr + expert_idx * stride_w_E + n_idx[:, None] * stride_w_N + k_idx[None, :] * stride_w_K
        gate_tile = tl.load(gate_ptrs, mask=weight_mask, other=0.0)
        gate_acc = tl.dot(x_tile, tl.trans(gate_tile), acc=gate_acc)

        up_ptrs = gate_ptrs + I_dim * stride_w_N
        up_tile = tl.load(up_ptrs, mask=weight_mask, other=0.0)
        up_acc = tl.dot(x_tile, tl.trans(up_tile), acc=up_acc)

    output_mask = row_mask[:, None] & n_mask[None, :]
    pre_gate_ptrs = pre_act_ptr + row_offs[:, None] * stride_pre_TK + n_idx[None, :] * stride_pre_N
    pre_up_ptrs = pre_gate_ptrs + I_dim * stride_pre_N
    tl.store(
        pre_gate_ptrs,
        gate_acc.to(pre_act_ptr.dtype.element_ty),
        mask=output_mask,
    )
    tl.store(
        pre_up_ptrs,
        up_acc.to(pre_act_ptr.dtype.element_ty),
        mask=output_mask,
    )

    output = gate_acc * tl.sigmoid(gate_acc) * up_acc
    output_ptrs = output_ptr + row_offs[:, None] * stride_output_TK + n_idx[None, :] * stride_output_N
    tl.store(
        output_ptrs,
        output.to(output_ptr.dtype.element_ty),
        mask=output_mask,
    )


def prepare(x, gate_weight, up_weight, expert_ids):
    del expert_ids
    tokens, hidden = x.shape
    experts, intermediate, _ = gate_weight.shape
    if tokens % experts != 0:
        raise ValueError("Triton comparison expects balanced blocked routing")
    rows_per_expert = tokens // experts

    gate_up = torch.cat((gate_weight, up_weight), dim=1).contiguous()
    x_gather_idx = torch.arange(
        tokens,
        dtype=torch.int32,
        device=x.device,
    )
    expert_start = (
        torch.arange(
            experts + 1,
            dtype=torch.int32,
            device=x.device,
        )
        * rows_per_expert
    )
    tile_row_start = torch.arange(
        0,
        tokens,
        BLOCK_M,
        dtype=torch.int32,
        device=x.device,
    )
    tile_expert = (tile_row_start // rows_per_expert).to(torch.int32)
    return {
        "x": x,
        "gate_up": gate_up,
        "x_gather_idx": x_gather_idx,
        "expert_start": expert_start,
        "tile_row_start": tile_row_start,
        "tile_expert": tile_expert,
        "pre_act": torch.empty(
            tokens,
            2 * intermediate,
            dtype=x.dtype,
            device=x.device,
        ),
        "output": torch.empty(
            tokens,
            intermediate,
            dtype=x.dtype,
            device=x.device,
        ),
        "hidden": hidden,
        "intermediate": intermediate,
    }


def launch(state):
    grid = lambda meta: (
        state["tile_row_start"].shape[0],
        triton.cdiv(state["intermediate"], meta["BLOCK_N"]),
    )
    fused_gate_up_swiglu_kernel[grid](
        state["x"],
        state["gate_up"],
        state["x_gather_idx"],
        state["expert_start"],
        state["tile_row_start"],
        state["tile_expert"],
        state["pre_act"],
        state["output"],
        H_dim=state["hidden"],
        I_dim=state["intermediate"],
        stride_x_T=state["x"].stride(0),
        stride_x_H=state["x"].stride(1),
        stride_w_E=state["gate_up"].stride(0),
        stride_w_N=state["gate_up"].stride(1),
        stride_w_K=state["gate_up"].stride(2),
        stride_pre_TK=state["pre_act"].stride(0),
        stride_pre_N=state["pre_act"].stride(1),
        stride_output_TK=state["output"].stride(0),
        stride_output_N=state["output"].stride(1),
        BLOCK_M=BLOCK_M,
    )
    return state["output"]
