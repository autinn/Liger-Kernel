"""CuTeDSL persistent-1CTA, AccStages=2 fused gate/up SwiGLU provider.

This is a comparison-local snapshot of
``Liger-Kernel/src/liger_kernel/ops/cutedsl/ops/fused_swiglu_gate_up.py``.
Its persistent scheduler follows NVIDIA CUTLASS's Blackwell CuTeDSL persistent
GEMM example:
https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py

The comparison changes only the epilogue reciprocal to ``rcp_approx`` so its
math contract matches CUDA and cuTile.

Computes the first phase of a (MoE) SwiGLU MLP in a single kernel::

    Z = SiLU(X @ W_gate^T) * (X @ W_up^T)

Both GEMMs contract over the hidden dim ``H`` and share the **same** ``X`` tile, so
X is TMA-loaded from gmem once per k-tile and consumed by two ``tcgen05`` UMMA
instructions writing two independent TMEM accumulators (``U`` and ``V``). The
epilogue reads both accumulators out of TMEM, applies ``SiLU(U) * V`` in registers,
converts to the output dtype and TMA-stores ``Z`` — the elementwise activation is
never round-tripped through gmem.

This is a genuine ``cutlass.cute`` implementation: it emits ``@cute.kernel`` device
code, compiles host-side with ``cute.compile`` (cached), marshals torch tensors via
DLPack and launches on torch's current CUDA stream.

Shapes / layouts::

    X          : (T, H)        row-major (H contiguous)          -> MMA A operand
    W_gate     : (E, I, H)     row-major (H contiguous)          -> MMA B operand
    W_up       : (E, I, H)     row-major (H contiguous)          -> MMA B operand
    Z          : (T, I)        row-major (I contiguous)
    expert_ids : (ceil(T/TileM),) int32 — expert routed to each TileM row-block

``E == 1`` (or ``expert_ids=None``) degenerates to the plain dense SwiGLU gate/up
projection, so the same kernel serves dense and MoE models.

Structure mirrors CUTLASS 4.5.2's ``blackwell/dense_gemm.py``, with four deltas:
  1. two B-operands (gate and up) sharing one A-operand and one TMA mbarrier,
  2. two TMEM accumulators allocated as a 2-"stage" fragment (U at 0, V at 1),
  3. a fused ``SiLU(U) * V`` epilogue instead of a plain type conversion,
  4. per-m-tile expert indexing into the L mode of the two weight tensors.

Requires an sm_100 (Blackwell) GPU.
"""

from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.gemm.sm100 as gemm_sm100
import torch

from cutlass.cute.nvgpu import cpasync
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive
from cutlass.pipeline import pipeline_init_wait

# Compiled-kernel cache keyed on everything the kernel bakes in (dtypes, tile
# config, static shapes). Without it every call would re-run ``cute.compile``.
_compile_cache = {}

# Cache the CUstream wrapper keyed on torch's raw stream handle so we don't
# rebuild the cuda.CUstream object every launch.
_stream_cache = {}


def to_cute_tensor(t, leading_dim=None, assumed_align=16):
    """Convert a torch tensor to a dynamically laid-out CuTe tensor."""
    ct = from_dlpack(t.detach(), assumed_align=assumed_align)
    ld = (t.ndim - 1) if leading_dim is None else leading_dim
    return ct.mark_layout_dynamic(leading_dim=ld)


def _cute_stream():
    raw = torch.cuda.current_stream().cuda_stream
    s = _stream_cache.get(raw)
    if s is None:
        s = cuda.CUstream(raw)
        _stream_cache[raw] = s
    return s


class FusedSwigluGateUpKernel:
    """Fused MoE MLP phase-1 (`Z = SiLU(X·Bᵀ)·(X·Cᵀ)`) for SM100.

    :param acc_dtype: accumulator type (Float32)
    :param use_2cta_instrs: use the cta_group=2 tcgen05 MMA variant
    :param mma_tiler_mn: (M, N) MMA tiler shape
    :param cluster_shape_mn: (ClusterM, ClusterN) CTA cluster shape
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
    ):
        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler_mn = mma_tiler_mn
        self.mma_tiler = (*mma_tiler_mn, 1)

        self.cta_group = tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE

        self.occupancy = 1
        self.threads_per_cta = 128
        # U and V accumulators live in TMEM as two "stages" of one fragment.
        self.num_acc_buf = 2

    # ── static setup ────────────────────────────────────────────────

    def _setup_attributes(self):
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.ab_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk,
            self.use_2cta_instrs,
            self.z_layout,
            self.z_dtype,
        )

        self.smem_capacity = utils.get_smem_capacity_in_bytes()

        self.num_ab_stage, self.num_z_stage = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.ab_dtype,
            self.epi_tile,
            self.z_dtype,
            self.z_layout,
            self.smem_capacity,
            self.occupancy,
        )

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.ab_dtype, self.num_ab_stage
        )
        # B and C are both MMA-B operands with identical shapes/majorness.
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.ab_dtype, self.num_ab_stage
        )
        self.z_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.z_dtype, self.z_layout, self.epi_tile, self.num_z_stage
        )

        # TMEM columns for BOTH accumulators (U and V).
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        acc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_buf))
        self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(acc_fake)

    # ── host entry ──────────────────────────────────────────────────

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        z: cute.Tensor,
        expert_ids: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.ab_dtype: Type[cutlass.Numeric] = x.element_type
        self.z_dtype: Type[cutlass.Numeric] = z.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(x).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.z_layout = utils.LayoutEnum.from_tensor(z)

        if cutlass.const_expr(x.element_type != b.element_type):
            raise TypeError(f"Type must match: {x.element_type} != {b.element_type}")
        if cutlass.const_expr(x.element_type != c.element_type):
            raise TypeError(f"Type must match: {x.element_type} != {c.element_type}")

        self._setup_attributes()

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.ab_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            x,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        tma_atom_c, tma_tensor_c = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            c,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        a_copy_size = cute.size_in_bytes(self.ab_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.ab_dtype, b_smem_layout)
        # One fused mbarrier per stage covering X + B + C (matches the C++ kernel).
        self.num_tma_load_bytes = (a_copy_size + 2 * b_copy_size) * atom_thr_size

        epi_smem_layout = cute.slice_(self.z_smem_layout_staged, (None, None, 0))
        tma_atom_z, tma_tensor_z = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            z,
            epi_smem_layout,
            self.epi_tile,
        )

        grid = self._compute_grid(z, self.cta_tile_shape_mnk, self.cluster_shape_mn)

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            tma_atom_z,
            tma_tensor_z,
            expert_ids,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.z_smem_layout_staged,
            self.epi_tile,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )
        return

    # ── device kernel ───────────────────────────────────────────────

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mX_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_nkl: cute.Tensor,
        tma_atom_z: cute.CopyAtom,
        mZ_mnl: cute.Tensor,
        expert_ids: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        z_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)
            cpasync.prefetch_descriptor(tma_atom_z)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)

        cta_coord = (bidx, bidy, bidz)
        mma_tile_coord_mnl = (
            cta_coord[0] // cute.size(tiled_mma.thr_id.shape),
            cta_coord[1],
            cta_coord[2],
        )
        tidx, _, _ = cute.arch.thread_idx()

        # Expert routed to this m-tile: selects the L slice of B and C.
        expert_id = cute.arch.make_warp_uniform(cutlass.Int32(expert_ids[mma_tile_coord_mnl[0]]))

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_producer)
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_per_cta)
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=1,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        acc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
        acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)

        tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=0, num_threads=self.threads_per_cta)
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        sZ = smem.allocate_tensor(
            element_type=self.z_dtype,
            layout=z_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=z_smem_layout_staged.inner,
        )
        sA = smem.allocate_tensor(
            element_type=self.ab_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.ab_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        sC = smem.allocate_tensor(
            element_type=self.ab_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )

        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        gX_mkl = cute.local_tile(mX_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gB_nkl = cute.local_tile(mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
        gC_nkl = cute.local_tile(mC_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
        gZ_mnl = cute.local_tile(mZ_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
        k_tile_cnt = cute.size(gX_mkl, mode=[3])

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gX_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_B(gC_nkl)
        tCgZ = thr_mma.partition_C(gZ_mnl)

        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tCsC, tCgC_tma = cpasync.tma_partition(
            tma_atom_c,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sC, 0, 3),
            cute.group_modes(tCgC, 0, 3),
        )

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        tCrC = tiled_mma.make_fragment_B(sC)

        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, 2) — stage 0 holds U = X·Bᵀ, stage 1 holds V = X·Cᵀ.
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_buf))

        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        tmem.allocate(self.num_tmem_alloc_cols)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
        tCtAccU = tCtAcc_base[(None, None, None, 0)]
        tCtAccV = tCtAcc_base[(None, None, None, 1)]

        # Slice to this CTA's tile. X has a single L slice; B/C are indexed by expert.
        tAgA = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
        tBgB = tBgB[(None, mma_tile_coord_mnl[1], None, expert_id)]
        tCgC_tma = tCgC_tma[(None, mma_tile_coord_mnl[1], None, expert_id)]

        prefetch_k_tile_cnt = cutlass.min(self.num_ab_stage - 2, k_tile_cnt)
        if warp_idx == 0:
            for k_tile_idx in cutlass.range(prefetch_k_tile_cnt, unroll=1):
                producer_handle = ab_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, k_tile_idx)],
                    tAsA[(None, producer_handle.index)],
                    tma_bar_ptr=producer_handle.barrier,
                    mcast_mask=a_full_mcast_mask,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, k_tile_idx)],
                    tBsB[(None, producer_handle.index)],
                    tma_bar_ptr=producer_handle.barrier,
                    mcast_mask=b_full_mcast_mask,
                )
                cute.copy(
                    tma_atom_c,
                    tCgC_tma[(None, k_tile_idx)],
                    tCsC[(None, producer_handle.index)],
                    tma_bar_ptr=producer_handle.barrier,
                    mcast_mask=b_full_mcast_mask,
                )

            peek_ab_full_status = cutlass.Boolean(False)
            if is_leader_cta:
                peek_ab_full_status = ab_consumer.try_wait()

            peek_ab_empty_status = ab_producer.try_acquire()

            for k_tile_idx in cutlass.range(k_tile_cnt):
                if k_tile_idx < k_tile_cnt - prefetch_k_tile_cnt:
                    producer_handle = ab_producer.acquire_and_advance(peek_ab_empty_status)
                    cute.copy(
                        tma_atom_a,
                        tAgA[(None, producer_handle.count)],
                        tAsA[(None, producer_handle.index)],
                        tma_bar_ptr=producer_handle.barrier,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB[(None, producer_handle.count)],
                        tBsB[(None, producer_handle.index)],
                        tma_bar_ptr=producer_handle.barrier,
                        mcast_mask=b_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_c,
                        tCgC_tma[(None, producer_handle.count)],
                        tCsC[(None, producer_handle.index)],
                        tma_bar_ptr=producer_handle.barrier,
                        mcast_mask=b_full_mcast_mask,
                    )

                if is_leader_cta:
                    consumer_handle = ab_consumer.wait_and_advance(peek_ab_full_status)

                    num_kblks = cute.size(tCrA, mode=[2])
                    for kblk_idx in cutlass.range(num_kblks, unroll_full=True):
                        kblk_crd = (None, None, kblk_idx, consumer_handle.index)
                        # U += X · Bᵀ ; V += X · Cᵀ — both read the same A fragment.
                        cute.gemm(tiled_mma, tCtAccU, tCrA[kblk_crd], tCrB[kblk_crd], tCtAccU)
                        cute.gemm(tiled_mma, tCtAccV, tCrA[kblk_crd], tCrC[kblk_crd], tCtAccV)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

                    consumer_handle.release()

                if k_tile_idx + 1 < k_tile_cnt - prefetch_k_tile_cnt:
                    peek_ab_empty_status = ab_producer.try_acquire()

                if k_tile_idx + 1 < k_tile_cnt and is_leader_cta:
                    peek_ab_full_status = ab_consumer.try_wait()

            if is_leader_cta:
                acc_pipeline.producer_commit(acc_producer_state)

        tmem.relinquish_alloc_permit()
        acc_pipeline.consumer_wait(acc_consumer_state)

        self.epilogue(
            tidx,
            warp_idx,
            mma_tile_coord_mnl,
            tma_atom_z,
            tCtAccU,
            tCtAccV,
            sZ,
            tCgZ,
            epi_tile,
        )

        pipeline.sync(barrier_id=1)
        tmem.free(tmem_ptr)

        if warp_idx == 0:
            ab_producer.tail()
        return

    # ── epilogue helpers ────────────────────────────────────────────

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAccU: cute.Tensor,
        tAccV: cute.Tensor,
        gZ_mnl: cute.Tensor,
        epi_tile: cute.Tile,
        use_2cta_instrs: Union[cutlass.Boolean, bool],
    ):
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            self.z_layout,
            self.z_dtype,
            self.acc_dtype,
            epi_tile,
            use_2cta_instrs,
        )
        tAccU_epi = cute.flat_divide(tAccU[((None, None), 0, 0)], epi_tile)
        tAccV_epi = cute.flat_divide(tAccV[((None, None), 0, 0)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAccU_epi[(None, None, 0, 0)])

        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAccU = thr_copy_t2r.partition_S(tAccU_epi)
        tTR_tAccV = thr_copy_t2r.partition_S(tAccV_epi)

        gZ_epi = cute.flat_divide(gZ_mnl[((None, None), 0, 0, None, None, None)], epi_tile)
        tTR_gZ = thr_copy_t2r.partition_D(gZ_epi)
        rmem_shape = tTR_gZ[(None, None, None, 0, 0, 0, 0, 0)].shape
        tTR_rU = cute.make_rmem_tensor(rmem_shape, self.acc_dtype)
        tTR_rV = cute.make_rmem_tensor(rmem_shape, self.acc_dtype)
        return tiled_copy_t2r, tTR_tAccU, tTR_tAccV, tTR_rU, tTR_rV

    def epilog_smem_copy_and_partition(
        self,
        tiled_copy_t2r: cute.TiledCopy,
        tTR_rZ: cute.Tensor,
        tidx: cutlass.Int32,
        sZ: cute.Tensor,
    ):
        copy_atom_r2s = sm100_utils.get_smem_store_op(self.z_layout, self.z_dtype, self.acc_dtype, tiled_copy_t2r)
        tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
        thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
        tRS_sZ = thr_copy_r2s.partition_D(sZ)
        tRS_rZ = tiled_copy_r2s.retile(tTR_rZ)
        return tiled_copy_r2s, tRS_rZ, tRS_sZ

    @cute.jit
    def epilogue(
        self,
        epi_tidx: cutlass.Int32,
        warp_idx: cutlass.Int32,
        mma_tile_coord_mnl: Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
        tma_atom_z: cute.CopyAtom,
        tCtAccU: cute.Tensor,
        tCtAccV: cute.Tensor,
        sZ: cute.Tensor,
        tCgZ: cute.Tensor,
        epi_tile: cute.Tile,
    ) -> None:
        """TMEM -> reg -> `SiLU(U)·V` -> smem -> TMA store."""
        (
            tiled_copy_t2r,
            tTR_tAccU,
            tTR_tAccV,
            tTR_rU,
            tTR_rV,
        ) = self.epilog_tmem_copy_and_partition(epi_tidx, tCtAccU, tCtAccV, tCgZ, epi_tile, self.use_2cta_instrs)
        tTR_tAccU = cute.group_modes(tTR_tAccU, 3, cute.rank(tTR_tAccU))
        tTR_tAccV = cute.group_modes(tTR_tAccV, 3, cute.rank(tTR_tAccV))

        tTR_rZ = cute.make_rmem_tensor(tTR_rU.shape, self.z_dtype)
        tiled_copy_r2s, tRS_rZ, tRS_sZ = self.epilog_smem_copy_and_partition(tiled_copy_t2r, tTR_rZ, epi_tidx, sZ)

        tCgZ_epi = cute.flat_divide(tCgZ[((None, None), 0, 0, None, None, None)], epi_tile)
        bSG_sZ, bSG_gZ = cpasync.tma_partition(
            tma_atom_z,
            0,
            cute.make_layout(1),
            cute.group_modes(sZ, 0, 2),
            cute.group_modes(tCgZ_epi, 0, 2),
        )
        bSG_gZ = bSG_gZ[(None, None, None, *mma_tile_coord_mnl)]
        bSG_gZ = cute.group_modes(bSG_gZ, 1, cute.rank(bSG_gZ))

        z_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_per_cta)
        z_pipeline = pipeline.PipelineTmaStore.create(num_stages=self.num_z_stage, producer_group=z_producer_group)

        subtile_cnt = cute.size(tTR_tAccU.shape, mode=[3])
        for subtile_idx in cutlass.range(subtile_cnt):
            cute.copy(tiled_copy_t2r, tTR_tAccU[(None, None, None, subtile_idx)], tTR_rU)
            cute.copy(tiled_copy_t2r, tTR_tAccV[(None, None, None, subtile_idx)], tTR_rV)

            u = tiled_copy_r2s.retile(tTR_rU).load()
            v = tiled_copy_r2s.retile(tTR_rV).load()
            # Match the CUDA and cuTile fast-math paths.
            sig = cute.arch.rcp_approx(1.0 + cute.math.exp(-u, fastmath=True))
            tRS_rZ.store((u * sig * v).to(self.z_dtype))

            z_buffer = subtile_idx % self.num_z_stage
            cute.copy(tiled_copy_r2s, tRS_rZ, tRS_sZ[(None, None, None, z_buffer)])
            cute.arch.fence_proxy("async.shared", space="cta")
            pipeline.sync(barrier_id=1)

            if warp_idx == 0:
                cute.copy(tma_atom_z, bSG_sZ[(None, z_buffer)], bSG_gZ[(None, subtile_idx)])
                z_pipeline.producer_commit()
                z_pipeline.producer_acquire()
            pipeline.sync(barrier_id=1)

        z_pipeline.producer_tail()

    # ── static config helpers ───────────────────────────────────────

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        ab_dtype: Type[cutlass.Numeric],
        epi_tile: cute.Tile,
        z_dtype: Type[cutlass.Numeric],
        z_layout: utils.LayoutEnum,
        smem_capacity: int,
        occupancy: int,
    ) -> Tuple[int, int]:
        """A/B stage count, accounting for the *two* B-operand buffers (B and C)."""
        num_z_stage = 2

        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, ab_dtype, 1)
        b_smem_layout_stage_one = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, ab_dtype, 1)
        z_smem_layout_stage_one = sm100_utils.make_smem_layout_epi(z_dtype, z_layout, epi_tile, 1)

        # One X buffer + TWO weight buffers (B and C) per stage.
        ab_bytes_per_stage = cute.size_in_bytes(ab_dtype, a_smem_layout_stage_one) + 2 * cute.size_in_bytes(
            ab_dtype, b_smem_layout_stage_one
        )
        mbar_helpers_bytes = 1024
        z_bytes_per_stage = cute.size_in_bytes(z_dtype, z_smem_layout_stage_one)
        z_bytes = z_bytes_per_stage * num_z_stage

        num_ab_stage = (smem_capacity - (occupancy + 1) * (mbar_helpers_bytes + z_bytes)) // ab_bytes_per_stage

        num_z_stage += (
            smem_capacity - ab_bytes_per_stage * num_ab_stage - (occupancy + 1) * (mbar_helpers_bytes + z_bytes)
        ) // ((occupancy + 1) * z_bytes_per_stage)
        return num_ab_stage, num_z_stage

    @staticmethod
    def _compute_grid(
        z: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
    ) -> Tuple[int, int, int]:
        cluster_shape_mnl = (*cluster_shape_mn, 1)
        return cute.round_up(
            (
                cute.ceil_div(z.layout.shape[0], cta_tile_shape_mnk[0]),
                cute.ceil_div(z.layout.shape[1], cta_tile_shape_mnk[1]),
                z.layout.shape[2],
            ),
            cluster_shape_mnl,
        )


class FusedSwigluGateUpPersistentKernel:
    """Persistent, accumulator-double-buffered ``Z = SiLU(X@Wg^T) * (X@Wu^T)`` for SM100.

    This is the pipelined counterpart of :class:`FusedSwigluGateUpKernel`, and the
    structural match to the hand-written CUTLASS C++ MLP1 kernel. Two changes carry
    the performance:

    1. **Persistent tile scheduling** — a CTA loops over many output tiles instead of
       computing exactly one, so the per-tile prologue (``tcgen05.alloc``, pipeline and
       cluster init) is paid once per CTA rather than once per tile.
    2. **``AccStages = 2``** — TMEM holds *four* accumulators (U and V for each of two
       stages), so the MMA warp can fill stage ``n+1`` while the epilogue warps drain
       stage ``n``. This is what removes the epilogue from the critical path.

    Because a 1-SM MMA accumulator is ``TileN`` columns wide and TMEM has only 512
    columns, 2 stages x (U, V) requires ``4 * TileN <= 512``, i.e. **TileN <= 128** —
    the same constraint the C++ kernel encodes as
    ``static_assert(AccStages * (2 * TileN) <= 512)``. Pass ``num_acc_stage=1`` to A/B
    test the same tile shape without double buffering.

    Warp specialization, mirroring the C++ kernel's split. With the default
    ``num_epi_warpgroups=2`` this is 10 warps / 320 threads:

    * warps 0-3 : epilogue warpgroup 0 — drains the low half of ``TileN``
    * warps 4-7 : epilogue warpgroup 1 — drains the high half of ``TileN``
    * warp 8    : MMA (tcgen05 issue only)
    * warp 9    : TMA producer (X, W_gate, W_up)

    The two epilogue warpgroups are the DSL counterpart of the C++ kernel's
    ``WG0 = warps 4..7 / WG1 = warps 8..11`` pair: each drains a disjoint half of the
    accumulator's N extent from the *same* TMEM stage, halving the serial epilogue
    latency that the MMA warp has to hide. This is legal because a ``tcgen05.ld``
    warp addresses the TMEM sub-partition selected by ``warp_id % 4``, so warps 4-7
    form a second correctly-aligned warpgroup, and TMEM reads are non-destructive so
    both groups may read the same stage concurrently.

    Pass ``num_epi_warpgroups=1`` to A/B against the original single-warpgroup
    epilogue (6 warps / 192 threads).
    """

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        num_acc_stage: int = 2,
        num_epi_warpgroups: int = 2,
    ):
        self.acc_dtype: Type[cutlass.Numeric] = acc_dtype
        self.use_2cta_instrs = use_2cta_instrs
        self.cluster_shape_mn = cluster_shape_mn
        self.mma_tiler_mn = mma_tiler_mn
        self.mma_tiler = (*mma_tiler_mn, 1)
        self.arch = "sm_100"
        self.cta_group = tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE
        self.occupancy = 1
        self.use_tma_store = True

        # Accumulator staging: `num_acc_stage` pipeline stages, each owning U and V.
        self.num_acc_stage = num_acc_stage
        self.num_acc_buf = num_acc_stage * 2

        if num_epi_warpgroups not in (1, 2):
            raise ValueError(f"num_epi_warpgroups must be 1 or 2, got {num_epi_warpgroups}")
        self.num_epi_wg = num_epi_warpgroups
        self.epilogue_warp_id = tuple(range(4 * num_epi_warpgroups))
        self.mma_warp_id = 4 * num_epi_warpgroups
        self.tma_warp_id = self.mma_warp_id + 1
        self.threads_per_cta = 32 * (len(self.epilogue_warp_id) + 2)
        # Epilogue warpgroup g rendezvouses on named barrier `epilog_sync_bar_id + g`,
        # so the ids must not collide with the TMEM alloc/dealloc barriers below.
        self.epilog_sync_bar_id = 4
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3

    def _create_tiled_mma(self):
        return sm100_utils.make_trivial_tiled_mma(
            self.ab_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )

    def _setup_attributes(self):
        tiled_mma = self._create_tiled_mma()

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = 4
        self.mma_tiler = (self.mma_tiler[0], self.mma_tiler[1], mma_inst_shape_k * mma_inst_tile_k)
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self.epi_tile = sm100_utils.compute_epilogue_tile_shape(
            self.cta_tile_shape_mnk, self.use_2cta_instrs, self.z_layout, self.z_dtype
        )
        self.smem_capacity = utils.get_smem_capacity_in_bytes()

        self.num_ab_stage, self.num_z_stage = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.ab_dtype,
            self.epi_tile,
            self.z_dtype,
            self.z_layout,
            self.smem_capacity,
            self.occupancy,
        )

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.ab_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.ab_dtype, self.num_ab_stage
        )
        # Each epilogue warpgroup owns a private, contiguous slice of the Z smem stages
        # so the groups never alias each other's TMA staging buffer. Round the stage
        # count down to a multiple of the group count.
        self.num_z_stage -= self.num_z_stage % self.num_epi_wg
        if self.num_z_stage < self.num_epi_wg:
            raise ValueError(
                f"need >= {self.num_epi_wg} Z smem stages for {self.num_epi_wg} epilogue "
                f"warpgroups, got {self.num_z_stage}"
            )
        # Each warpgroup drains a disjoint set of epilogue subtiles along N.
        self.num_epi_subtiles = self.cta_tile_shape_mnk[1] // cute.size(self.epi_tile[1])
        if self.num_epi_subtiles % self.num_epi_wg != 0:
            raise ValueError(
                f"{self.num_epi_subtiles} epilogue subtiles do not split evenly across "
                f"{self.num_epi_wg} epilogue warpgroups"
            )

        self.z_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.z_dtype, self.z_layout, self.epi_tile, self.num_z_stage
        )

        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        acc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_buf))
        self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(acc_fake)
        if self.num_tmem_alloc_cols > 512:
            raise ValueError(
                f"TMEM overflow: {self.num_acc_stage} acc stages x (U,V) at TileN="
                f"{self.cta_tile_shape_mnk[1]} needs {self.num_tmem_alloc_cols} of 512 "
                f"columns. Use TileN <= {512 // (2 * self.num_acc_stage)} or num_acc_stage=1."
            )

        # Aliases so the stock CUTLASS sm100 epilogue helpers can be reused verbatim.
        self.c_dtype = self.z_dtype
        self.c_layout = self.z_layout

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        z: cute.Tensor,
        expert_ids: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.ab_dtype: Type[cutlass.Numeric] = x.element_type
        self.z_dtype: Type[cutlass.Numeric] = z.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(x).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.z_layout = utils.LayoutEnum.from_tensor(z)

        if cutlass.const_expr(x.element_type != b.element_type):
            raise TypeError(f"Type must match: {x.element_type} != {b.element_type}")
        if cutlass.const_expr(x.element_type != c.element_type):
            raise TypeError(f"Type must match: {x.element_type} != {c.element_type}")

        self._setup_attributes()
        tiled_mma = self._create_tiled_mma()
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, x, a_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )

        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b, b_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )
        tma_atom_c, tma_tensor_c = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, c, b_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )

        a_copy_size = cute.size_in_bytes(self.ab_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.ab_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + 2 * b_copy_size) * atom_thr_size

        epi_smem_layout = cute.slice_(self.z_smem_layout_staged, (None, None, 0))
        tma_atom_z, tma_tensor_z = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), z, epi_smem_layout, self.epi_tile
        )

        tile_sched_params, grid = self._compute_grid(
            z, self.cta_tile_shape_mnk, self.cluster_shape_mn, max_active_clusters
        )

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            tma_atom_z,
            tma_tensor_z,
            expert_ids,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.z_smem_layout_staged,
            self.epi_tile,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mX_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_nkl: cute.Tensor,
        tma_atom_z: cute.CopyAtom,
        mZ_mnl: cute.Tensor,
        expert_ids: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        z_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)
            cpasync.prefetch_descriptor(tma_atom_z)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_tma_producer)
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilogue_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, num_acc_consumer_threads)
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)

        sA = smem.allocate_tensor(
            element_type=self.ab_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.ab_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        sC = smem.allocate_tensor(
            element_type=self.ab_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )

        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        gX_mkl = cute.local_tile(mX_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gB_nkl = cute.local_tile(mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
        gC_nkl = cute.local_tile(mC_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
        gZ_mnl = cute.local_tile(mZ_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))
        k_tile_cnt = cute.size(gX_mkl, mode=[3])

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gX_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_B(gC_nkl)
        tCgZ = thr_mma.partition_C(gZ_mnl)

        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tCsC, tCgC_tma = cpasync.tma_partition(
            tma_atom_c,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sC, 0, 3),
            cute.group_modes(tCgC, 0, 3),
        )

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        tCrC = tiled_mma.make_fragment_B(sC)

        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        # (MMA, MMA_M, MMA_N, 2*STAGE): buffer 2s = U, buffer 2s+1 = V for stage s.
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_buf))

        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        #
        # Specialized TMA load warp — streams X, W_gate and W_up on one mbarrier
        #
        if warp_idx == self.tma_warp_id:
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                # Expert routed to this MMA-tile row block selects the L slice of Wg/Wu.
                expert_id = cutlass.Int32(expert_ids[mma_tile_coord_mnl[0]])

                tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, expert_id)]
                tCgC_slice = tCgC_tma[(None, mma_tile_coord_mnl[1], None, expert_id)]

                ab_producer.reset()
                peek_ab_empty_status = ab_producer.try_acquire()

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    handle = ab_producer.acquire_and_advance(peek_ab_empty_status)
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, handle.count)],
                        tAsA[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, handle.count)],
                        tBsB[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=b_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_c,
                        tCgC_slice[(None, handle.count)],
                        tCsC[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=b_full_mcast_mask,
                    )
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek_ab_empty_status = ab_producer.try_acquire()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            ab_producer.tail()

        #
        # Specialized MMA warp — issues both UMMAs into the current acc stage
        #
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            acc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_acc_stage)

            while work_tile.is_valid_tile:
                acc_buf = acc_producer_state.index * 2
                tCtAccU = tCtAcc_base[(None, None, None, acc_buf)]
                tCtAccV = tCtAcc_base[(None, None, None, acc_buf + 1)]

                ab_consumer.reset()
                peek_ab_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    peek_ab_full_status = ab_consumer.try_wait()

                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                for k_tile in range(k_tile_cnt):
                    if is_leader_cta:
                        handle = ab_consumer.wait_and_advance(peek_ab_full_status)
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblk_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblk_crd = (None, None, kblk_idx, handle.index)
                            # U += X.Wg^T and V += X.Wu^T share the same A fragment.
                            cute.gemm(tiled_mma, tCtAccU, tCrA[kblk_crd], tCrB[kblk_crd], tCtAccU)
                            cute.gemm(tiled_mma, tCtAccV, tCrA[kblk_crd], tCrC[kblk_crd], tCtAccV)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        handle.release()
                        peek_ab_full_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_full_status = ab_consumer.try_wait()

                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            acc_pipeline.producer_tail(acc_producer_state)

        # (EPI_TILE_M, EPI_TILE_N, STAGE)
        sZ = smem.allocate_tensor(
            element_type=self.z_dtype,
            layout=z_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=z_smem_layout_staged.inner,
        )

        #
        # Specialized epilogue warps — drain stage n while MMA fills stage n+1
        #
        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_acc_stage)
            # One TMA-store pipeline per epilogue warpgroup: PipelineTmaStore is
            # mbarrier-free (it is a cp.async.bulk commit/wait_group scoreboard), so
            # each group's leader warp tracks only its own in-flight stores.
            z_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 128)
            z_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.num_z_stage // self.num_epi_wg, producer_group=z_producer_group
            )

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                num_tiles_executed = tile_sched.num_tiles_executed

                acc_consumer_state = self.epilogue(
                    tidx,
                    warp_idx,
                    tma_atom_z,
                    tCtAcc_base,
                    sZ,
                    tCgZ,
                    epi_tile,
                    num_tiles_executed,
                    mma_tile_coord_mnl,
                    acc_consumer_state,
                    acc_pipeline,
                    z_pipeline,
                )

            z_pipeline.producer_tail()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

    @cute.jit
    def epilogue(
        self,
        epi_tidx: cutlass.Int32,
        warp_idx: cutlass.Int32,
        tma_atom_z: cute.CopyAtom,
        tCtAcc_base: cute.Tensor,
        sZ: cute.Tensor,
        tCgZ_base: cute.Tensor,
        epi_tile: cute.Tile,
        num_tiles_executed: cutlass.Int32,
        mma_tile_coord_mnl: Tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32],
        acc_consumer_state: pipeline.PipelineState,
        acc_pipeline: pipeline.PipelineAsync,
        z_pipeline: pipeline.PipelineTmaStore,
    ) -> pipeline.PipelineState:
        """TMEM(U,V) -> reg -> ``SiLU(U)*V`` -> smem -> TMA store, for one output tile.

        With ``num_epi_warpgroups=2`` this runs on both aligned epilogue warpgroups at
        once: warpgroup ``g`` drains subtiles ``[g*S, (g+1)*S)`` of the N extent, where
        ``S = num_epi_subtiles / num_epi_wg``. Both groups read the *same* TMEM
        accumulator stage (TMEM reads are non-destructive) but disjoint column ranges,
        and stage through disjoint Z smem buffers, so they never interfere.
        """
        # Warpgroup identity. `warp_idx` is warp-uniform, so `epi_wg` is too — which is
        # required because it feeds a TMEM column offset.
        epi_wg = warp_idx // 4
        # tcgen05.ld / the r2s tiled copy are warpgroup-collective and expect a
        # thread id in [0, 128).
        epi_tidx = epi_tidx % 128

        tCgZ = gemm_sm100.transform_partitioned_tensor_layout(tCgZ_base)
        tCtAcc = gemm_sm100.transform_partitioned_tensor_layout(tCtAcc_base)

        tiled_copy_t2r, tTR_tAcc_base, tTR_rU = gemm_sm100.epilogue_tmem_copy_and_partition(
            self, epi_tidx, tCtAcc, tCgZ, epi_tile, self.use_2cta_instrs
        )
        tTR_rV = cute.make_rmem_tensor(tTR_rU.shape, self.acc_dtype)
        tTR_rZ = cute.make_rmem_tensor(tTR_rU.shape, self.z_dtype)
        tiled_copy_r2s, tRS_rZ, tRS_sZ = gemm_sm100.epilogue_smem_copy_and_partition(
            self, tiled_copy_t2r, tTR_rZ, epi_tidx, sZ
        )

        tCgZ_epi = cute.flat_divide(tCgZ, epi_tile)
        bSG_sZ, bSG_gZ_partitioned = cpasync.tma_partition(
            tma_atom_z,
            0,
            cute.make_layout(1),
            cute.group_modes(sZ, 0, 2),
            cute.group_modes(tCgZ_epi, 0, 2),
        )

        epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=self.epilog_sync_bar_id + epi_wg,
            num_threads=128,
        )

        bSG_gZ = bSG_gZ_partitioned[(None, None, None, *mma_tile_coord_mnl)]

        acc_buf = acc_consumer_state.index * 2
        tTR_tAccU = tTR_tAcc_base[(None, None, None, None, None, acc_buf)]
        tTR_tAccV = tTR_tAcc_base[(None, None, None, None, None, acc_buf + 1)]

        acc_pipeline.consumer_wait(acc_consumer_state)

        tTR_tAccU = cute.group_modes(tTR_tAccU, 3, cute.rank(tTR_tAccU))
        tTR_tAccV = cute.group_modes(tTR_tAccV, 3, cute.rank(tTR_tAccV))
        bSG_gZ = cute.group_modes(bSG_gZ, 1, cute.rank(bSG_gZ))

        subtile_cnt = cute.size(tTR_tAccU.shape, mode=[3])
        # Split the N extent across the epilogue warpgroups: group `epi_wg` owns
        # subtiles [epi_wg*sub_per_wg, +sub_per_wg) and Z smem stages
        # [epi_wg*z_per_wg, +z_per_wg).
        sub_per_wg = subtile_cnt // self.num_epi_wg
        z_per_wg = self.num_z_stage // self.num_epi_wg
        wg_sub_base = epi_wg * sub_per_wg
        wg_z_base = epi_wg * z_per_wg
        num_prev_subtiles = num_tiles_executed * sub_per_wg
        for local_idx in range(sub_per_wg):
            subtile_idx = wg_sub_base + local_idx
            cute.copy(tiled_copy_t2r, tTR_tAccU[(None, None, None, subtile_idx)], tTR_rU)
            cute.copy(tiled_copy_t2r, tTR_tAccV[(None, None, None, subtile_idx)], tTR_rV)

            tRS_rU = tiled_copy_r2s.retile(tTR_rU)
            tRS_rV = tiled_copy_r2s.retile(tTR_rV)
            for i in cutlass.range_constexpr(cute.size(tRS_rZ)):
                u = tRS_rU[i].to(cutlass.Float32)
                v = tRS_rV[i].to(cutlass.Float32)
                sig = cute.arch.rcp_approx(1.0 + cute.math.exp(-u, fastmath=True))
                tRS_rZ[i] = (u * sig * v).to(self.z_dtype)

            z_buffer = wg_z_base + (num_prev_subtiles + local_idx) % z_per_wg
            cute.copy(tiled_copy_r2s, tRS_rZ, tRS_sZ[(None, None, None, z_buffer)])
            cute.arch.fence_proxy("async.shared", space="cta")
            epilog_sync_barrier.arrive_and_wait()

            # Leader warp of *this* warpgroup issues the store (warps 0 and 4).
            if warp_idx % 4 == 0:
                cute.copy(tma_atom_z, bSG_sZ[(None, z_buffer)], bSG_gZ[(None, subtile_idx)])
                z_pipeline.producer_commit()
                z_pipeline.producer_acquire()
            epilog_sync_barrier.arrive_and_wait()

        epilog_sync_barrier.arrive_and_wait()

        # One arrival per epilogue warp (elect_one elects a single lane per warp), so
        # the accumulator stage is released only once every warp of every epilogue
        # warpgroup has finished reading it. This contributes
        # `len(self.epilogue_warp_id)` arrivals per CTA; the pipeline doubles that
        # count for a two-CTA cluster.
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(acc_consumer_state)
        acc_consumer_state.advance()
        return acc_consumer_state

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        ab_dtype: Type[cutlass.Numeric],
        epi_tile: cute.Tile,
        z_dtype: Type[cutlass.Numeric],
        z_layout: utils.LayoutEnum,
        smem_capacity: int,
        occupancy: int,
    ) -> Tuple[int, int]:
        """A/B stage count, accounting for the *two* weight buffers (gate and up)."""
        num_z_stage = 2

        a_one = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, ab_dtype, 1)
        b_one = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, ab_dtype, 1)
        z_one = sm100_utils.make_smem_layout_epi(z_dtype, z_layout, epi_tile, 1)

        ab_bytes_per_stage = cute.size_in_bytes(ab_dtype, a_one) + 2 * cute.size_in_bytes(ab_dtype, b_one)
        mbar_helpers_bytes = 1024
        z_bytes_per_stage = cute.size_in_bytes(z_dtype, z_one)
        z_bytes = z_bytes_per_stage * num_z_stage

        num_ab_stage = (smem_capacity // occupancy - (mbar_helpers_bytes + z_bytes)) // ab_bytes_per_stage

        num_z_stage += (
            smem_capacity - occupancy * ab_bytes_per_stage * num_ab_stage - occupancy * (mbar_helpers_bytes + z_bytes)
        ) // (occupancy * z_bytes_per_stage)
        return num_ab_stage, num_z_stage

    @staticmethod
    def _compute_grid(
        z: cute.Tensor,
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        max_active_clusters: cutlass.Constexpr,
    ):
        z_shape = cute.slice_(cta_tile_shape_mnk, (None, None, 0))
        gz = cute.zipped_divide(z, tiler=z_shape)
        num_ctas_mnl = gz[(0, (None, None, None))].shape
        cluster_shape_mnl = (*cluster_shape_mn, 1)
        tile_sched_params = utils.PersistentTileSchedulerParams(num_ctas_mnl, cluster_shape_mnl)
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(tile_sched_params, max_active_clusters)
        return tile_sched_params, grid


# ═══════════════════════════════════════════════════════════════════
# Host-side entry points
# ═══════════════════════════════════════════════════════════════════

_TORCH_TO_CUTLASS_DTYPE = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
}

_max_active_clusters_cache = {}


def _max_active_clusters(cluster_size: int) -> int:
    n = _max_active_clusters_cache.get(cluster_size)
    if n is None:
        n = utils.HardwareInfo().get_max_active_clusters(cluster_size)
        _max_active_clusters_cache[cluster_size] = n
    return n


def _as_mkl(t: torch.Tensor) -> torch.Tensor:
    """(M, K) -> (M, K, 1) with a well-formed L stride (M*K), not a degenerate 1."""
    return t.unsqueeze(0).permute(1, 2, 0)


def _as_nkl(w: torch.Tensor) -> torch.Tensor:
    """(E, I, H) -> (I, H, E): N=I, K=H (contiguous), L=E."""
    return w.permute(1, 2, 0)


def expert_block_size(mma_tiler_mn: Tuple[int, int] = (256, 128)) -> int:
    """Rows of X that share one expert id.

    The kernel routes at **MMA-tile** granularity, so an ``expert_ids`` entry covers
    ``mma_tiler_mn[0]`` token rows — 256 for a 2-CTA MMA, 128 for 1-CTA. Callers and
    the reference must agree on this or the comparison is meaningless.
    """
    return mma_tiler_mn[0]


def fused_swiglu_gate_up_forward(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    expert_ids: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    mma_tiler_mn: Tuple[int, int] = (256, 128),
    cluster_shape_mn: Tuple[int, int] = (2, 1),
    use_2cta_instrs: bool = True,
    persistent: bool = True,
    num_acc_stage: int = 2,
    num_epi_warpgroups: int = 2,
):
    """``Z = SiLU(X @ W_gate^T) * (X @ W_up^T)`` on Blackwell via CuTe DSL.

    :param x: ``(T, H)`` activations, bf16/fp16, row-major.
    :param w_gate: ``(E, I, H)`` or ``(I, H)`` gate weights, row-major.
    :param w_up: ``(E, I, H)`` or ``(I, H)`` up weights, row-major.
    :param expert_ids: ``(ceil(T / expert_block_size(mma_tiler_mn)),)`` int32 expert per
        MMA-tile row block. ``None`` routes every block to expert 0.
    :param out: optional ``(T, I)`` output buffer.
    :param persistent: use the persistent, accumulator-double-buffered kernel
        (:class:`FusedSwigluGateUpPersistentKernel`). ``False`` selects the simpler
        one-tile-per-CTA :class:`FusedSwigluGateUpKernel`.
    :param num_acc_stage: TMEM accumulator stages (persistent kernel only). ``2`` overlaps
        the epilogue of tile *n* with the MMA of tile *n+1*; requires ``TileN <= 128``.
    :param num_epi_warpgroups: epilogue warpgroups (persistent kernel only). ``2`` mirrors
        the C++ kernel's two aligned epilogue warpgroups, halving the serial epilogue
        latency; ``1`` selects the original single-warpgroup epilogue.
    :returns: ``(T, I)`` tensor of the same dtype as ``x``.
    """
    if x.ndim != 2:
        raise ValueError(f"x must be 2-D (T, H), got {tuple(x.shape)}")
    if w_gate.ndim == 2:
        w_gate = w_gate.unsqueeze(0)
    if w_up.ndim == 2:
        w_up = w_up.unsqueeze(0)
    if w_gate.shape != w_up.shape:
        raise ValueError(f"w_gate {tuple(w_gate.shape)} != w_up {tuple(w_up.shape)}")

    dtype = x.dtype
    if dtype not in _TORCH_TO_CUTLASS_DTYPE:
        raise ValueError(f"unsupported dtype {dtype}; use bfloat16 or float16")
    if w_gate.dtype != dtype or w_up.dtype != dtype:
        raise ValueError("x, w_gate and w_up must share a dtype")

    t_dim, h_dim = x.shape
    e_dim, i_dim, h_w = w_gate.shape
    if h_w != h_dim:
        raise ValueError(f"hidden dim mismatch: x has {h_dim}, weights have {h_w}")

    blk = expert_block_size(mma_tiler_mn)
    num_expert_blocks = (t_dim + blk - 1) // blk
    if expert_ids is None:
        expert_ids = torch.zeros(num_expert_blocks, dtype=torch.int32, device=x.device)
    else:
        expert_ids = expert_ids.to(device=x.device, dtype=torch.int32).contiguous()
        if expert_ids.numel() < num_expert_blocks:
            raise ValueError(
                f"expert_ids needs >= {num_expert_blocks} entries (one per {blk}-row block), got {expert_ids.numel()}"
            )

    if out is None:
        out = torch.empty(t_dim, i_dim, dtype=dtype, device=x.device)

    x_ct = to_cute_tensor(_as_mkl(x), leading_dim=1)
    g_ct = to_cute_tensor(_as_nkl(w_gate), leading_dim=1)
    u_ct = to_cute_tensor(_as_nkl(w_up), leading_dim=1)
    z_ct = to_cute_tensor(_as_mkl(out), leading_dim=1)
    e_ct = to_cute_tensor(expert_ids, leading_dim=0)

    stream = _cute_stream()
    key = (
        dtype,
        t_dim,
        h_dim,
        i_dim,
        e_dim,
        mma_tiler_mn,
        cluster_shape_mn,
        use_2cta_instrs,
        persistent,
        num_acc_stage,
        num_epi_warpgroups,
    )
    if key not in _compile_cache:
        if persistent:
            kernel = FusedSwigluGateUpPersistentKernel(
                acc_dtype=cutlass.Float32,
                use_2cta_instrs=use_2cta_instrs,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                num_acc_stage=num_acc_stage,
                num_epi_warpgroups=num_epi_warpgroups,
            )
            mac = _max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1])
            _compile_cache[key] = cute.compile(kernel, x_ct, g_ct, u_ct, z_ct, e_ct, mac, stream)
            _compile_cache[key](x_ct, g_ct, u_ct, z_ct, e_ct, stream)
            return out
        kernel = FusedSwigluGateUpKernel(
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
        )
        _compile_cache[key] = cute.compile(kernel, x_ct, g_ct, u_ct, z_ct, e_ct, stream)

    _compile_cache[key](x_ct, g_ct, u_ct, z_ct, e_ct, stream)
    return out


def fused_swiglu_gate_up_reference(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    expert_ids: Optional[torch.Tensor] = None,
    tile_m: int = 256,
) -> torch.Tensor:
    """fp32 torch reference for :func:`fused_swiglu_gate_up_forward`.

    ``tile_m`` must equal :func:`expert_block_size` for the kernel config under test.
    """
    if w_gate.ndim == 2:
        w_gate = w_gate.unsqueeze(0)
    if w_up.ndim == 2:
        w_up = w_up.unsqueeze(0)
    t_dim = x.shape[0]
    i_dim = w_gate.shape[1]
    num_blocks = (t_dim + tile_m - 1) // tile_m
    if expert_ids is None:
        expert_ids = torch.zeros(num_blocks, dtype=torch.int32, device=x.device)

    xf = x.float()
    out = torch.empty(t_dim, i_dim, dtype=torch.float32, device=x.device)
    for m in range(num_blocks):
        lo, hi = m * tile_m, min((m + 1) * tile_m, t_dim)
        e = int(expert_ids[m])
        xs = xf[lo:hi]
        u = xs @ w_gate[e].float().T
        v = xs @ w_up[e].float().T
        out[lo:hi] = torch.nn.functional.silu(u) * v
    return out.to(x.dtype)


def prepare(x, gate_weight, up_weight, expert_ids):
    """Allocate the output and package one benchmark invocation."""
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
    """Launch the production-pipeline-shaped 1CTA fast-math kernel."""
    return fused_swiglu_gate_up_forward(
        state["x"],
        state["gate_weight"],
        state["up_weight"],
        expert_ids=state["expert_ids"],
        out=state["output"],
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
        use_2cta_instrs=False,
        persistent=True,
        num_acc_stage=2,
        num_epi_warpgroups=2,
    )
