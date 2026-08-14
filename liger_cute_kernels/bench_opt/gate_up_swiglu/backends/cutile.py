"""cuTile persistent-1CTA fused gate/up SwiGLU provider.

Written for this comparison using NVIDIA's official cuTile MatMul and MoE
samples as the API and tiling references:
https://github.com/NVIDIA/cutile-python/blob/main/samples/MatMul.py
https://github.com/NVIDIA/cutile-python/blob/main/samples/MoE.py

cuTile/TileIRAS does not expose a TMEM accumulator-stage setting. This backend
matches the production CUDA tile and CTA topology, while pipeline and
accumulator staging remain compiler-managed.
"""

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]

TILE_M = 128
TILE_N = 128
TILE_K = 64
ROWS_PER_EXPERT_ID = 128
PERSISTENT_BLOCKS = 148


@ct.kernel(num_ctas=ct.ByTarget(sm_100=1), occupancy=1)
def cutile_gate_up_swiglu_kernel(
    x,
    gate_weight,
    up_weight,
    output,
    expert_ids,
    tile_m: ConstInt,
    tile_n: ConstInt,
    tile_k: ConstInt,
    rows_per_expert_id: ConstInt,
):
    """Compute both GEMMs and the SwiGLU epilogue in one persistent kernel."""
    block = ct.bid(0)
    num_m_tiles = ct.cdiv(x.shape[0], tile_m)
    num_n_tiles = ct.cdiv(output.shape[1], tile_n)
    num_k_tiles = ct.cdiv(x.shape[1], tile_k)
    num_output_tiles = num_m_tiles * num_n_tiles
    num_blocks = ct.num_blocks(0)
    zero_pad = ct.PaddingMode.ZERO

    for linear in range(block, num_output_tiles, num_blocks):
        m_tile = linear % num_m_tiles
        n_tile = linear // num_m_tiles
        expert_tile = (m_tile * tile_m) // rows_per_expert_id
        expert_id = ct.load(expert_ids, index=expert_tile, shape=())

        gate_acc = ct.full((tile_m, tile_n), 0.0, dtype=ct.float32)
        up_acc = ct.full((tile_m, tile_n), 0.0, dtype=ct.float32)

        for k_tile in range(num_k_tiles):
            x_tile = ct.load(
                x,
                index=(m_tile, k_tile),
                shape=(tile_m, tile_k),
                padding_mode=zero_pad,
                latency=10,
            )
            gate_tile = ct.load(
                gate_weight,
                index=(expert_id, k_tile, n_tile),
                shape=(1, tile_k, tile_n),
                order=(0, 2, 1),
                padding_mode=zero_pad,
                latency=10,
            ).reshape((tile_k, tile_n))
            up_tile = ct.load(
                up_weight,
                index=(expert_id, k_tile, n_tile),
                shape=(1, tile_k, tile_n),
                order=(0, 2, 1),
                padding_mode=zero_pad,
                latency=10,
            ).reshape((tile_k, tile_n))

            gate_acc = ct.mma(x_tile, gate_tile, gate_acc)
            up_acc = ct.mma(x_tile, up_tile, up_acc)

        denominator = ct.add(1.0, ct.exp(-gate_acc), flush_to_zero=True)
        sigmoid = ct.truediv(
            1.0,
            denominator,
            flush_to_zero=True,
            rounding_mode=ct.RoundingMode.APPROX,
        )
        result = ct.mul(
            ct.mul(gate_acc, sigmoid, flush_to_zero=True),
            up_acc,
            flush_to_zero=True,
        )
        ct.store(
            output,
            index=(m_tile, n_tile),
            tile=ct.astype(result, output.dtype),
        )


def prepare(x, gate_weight, up_weight, expert_ids):
    return {
        "x": x,
        "gate_weight": gate_weight,
        "up_weight": up_weight,
        "expert_ids": expert_ids,
        "output": torch.empty(
            x.shape[0],
            gate_weight.shape[1],
            dtype=x.dtype,
            device=x.device,
        ),
    }


def launch(state):
    ct.launch(
        torch.cuda.current_stream(state["x"].device),
        (PERSISTENT_BLOCKS, 1, 1),
        cutile_gate_up_swiglu_kernel,
        (
            state["x"],
            state["gate_weight"],
            state["up_weight"],
            state["output"],
            state["expert_ids"],
            TILE_M,
            TILE_N,
            TILE_K,
            ROWS_PER_EXPERT_ID,
        ),
    )
    return state["output"]
