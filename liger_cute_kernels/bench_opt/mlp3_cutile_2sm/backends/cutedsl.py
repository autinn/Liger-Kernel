"""CuTe DSL SM100 2CTA MLP3 weight-gradient backend.

Computes, for each expert:

    dA[expert] = dY[expert].T @ Z[expert]

The implementation matches the CUDA/CuTe experiment's logical
``256x256x64`` tile, persistent outer-split and split-K scheduler, paired-CTA
TMA/UMMA path, FP32 TMEM accumulation, and BF16 TMA reduction-add output.
"""

from typing import Tuple
from typing import Type

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

TILE_M = 256
TILE_N = 256
TILE_K = 64

_compile_cache = {}
_compile_metadata = {}
_stream_cache = {}
_max_active_clusters_cache = {}


def _to_cute_tensor(tensor, leading_dim, assumed_align=16):
    result = from_dlpack(tensor.detach(), assumed_align=assumed_align)
    return result.mark_layout_dynamic(leading_dim=leading_dim)


def _cute_stream():
    raw = torch.cuda.current_stream().cuda_stream
    stream = _stream_cache.get(raw)
    if stream is None:
        stream = cuda.CUstream(raw)
        _stream_cache[raw] = stream
    return stream


def _max_active_clusters(cluster_size):
    count = _max_active_clusters_cache.get(cluster_size)
    if count is None:
        count = utils.HardwareInfo().get_max_active_clusters(cluster_size)
        _max_active_clusters_cache[cluster_size] = count
    return count


def _as_mkl(tensor):
    """Convert a rank-2 tensor to an M/K/L=1 view without changing strides."""
    return tensor.unsqueeze(0).permute(1, 2, 0)


def _as_mnl(output):
    """Convert contiguous E/H/I output to H/I/E."""
    return output.permute(1, 2, 0)


class Mlp3TwoSmPersistentKernel:
    """Persistent two-CTA BF16 MLP3 with an explicit CuTe DSL pipeline."""

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        outer_split: int,
        k_split: int,
        num_experts: int,
        num_m_tiles: int,
        num_n_tiles: int,
        num_pairs: int,
        num_acc_stage: int = 2,
        num_epi_warpgroups: int = 2,
    ):
        self.acc_dtype = acc_dtype
        self.outer_split = outer_split
        self.k_split = k_split
        self.num_experts = num_experts
        self.num_m_tiles = num_m_tiles
        self.num_n_tiles = num_n_tiles
        self.num_pairs = num_pairs
        self.total_cells = num_experts * num_n_tiles * outer_split * k_split

        self.use_2cta_instrs = True
        self.cluster_shape_mn = (2, 1)
        self.mma_tiler = (TILE_M, TILE_N, TILE_K)
        self.cta_group = tcgen05.CtaGroup.TWO
        self.occupancy = 1
        self.num_acc_stage = num_acc_stage

        if num_epi_warpgroups not in (1, 2):
            raise ValueError(f"num_epi_warpgroups must be 1 or 2, got {num_epi_warpgroups}")
        self.num_epi_wg = num_epi_warpgroups
        self.epilogue_warp_id = tuple(range(4 * num_epi_warpgroups))
        self.mma_warp_id = 4 * num_epi_warpgroups
        self.tma_warp_id = self.mma_warp_id + 1
        self.threads_per_cta = 32 * (len(self.epilogue_warp_id) + 2)

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
            self.c_layout,
            self.c_dtype,
        )
        self.smem_capacity = utils.get_smem_capacity_in_bytes()
        self.num_ab_stage, self.num_c_stage = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.ab_dtype,
            self.epi_tile,
            self.c_dtype,
            self.c_layout,
            self.smem_capacity,
            self.occupancy,
        )
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.ab_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.ab_dtype,
            self.num_ab_stage,
        )

        self.num_c_stage -= self.num_c_stage % self.num_epi_wg
        if self.num_c_stage < self.num_epi_wg:
            raise ValueError(f"need at least {self.num_epi_wg} epilogue stages, got {self.num_c_stage}")
        self.num_epi_subtiles = self.cta_tile_shape_mnk[1] // cute.size(self.epi_tile[1])
        if self.num_epi_subtiles % self.num_epi_wg:
            raise ValueError(
                f"{self.num_epi_subtiles} epilogue subtiles do not divide across {self.num_epi_wg} warpgroups"
            )
        self.c_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.c_dtype,
            self.c_layout,
            self.epi_tile,
            self.num_c_stage,
        )
        self.mainloop_smem_bytes = cute.size_in_bytes(
            self.ab_dtype,
            self.a_smem_layout_staged,
        ) + cute.size_in_bytes(
            self.ab_dtype,
            self.b_smem_layout_staged,
        )
        self.epilogue_smem_bytes = cute.size_in_bytes(
            self.c_dtype,
            self.c_smem_layout_staged,
        )
        self.smem_budget_bytes = self.mainloop_smem_bytes + self.epilogue_smem_bytes + 1024

        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        acc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))
        self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(acc_fake)
        if self.num_tmem_alloc_cols > 512:
            raise ValueError(f"TMEM overflow: {self.num_tmem_alloc_cols} columns required")

    @cute.jit
    def __call__(
        self,
        dy_t: cute.Tensor,
        z_t: cute.Tensor,
        output: cute.Tensor,
        expert_k_starts: cute.Tensor,
        expert_k_ends: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.ab_dtype = dy_t.element_type
        self.c_dtype = output.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(dy_t).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(z_t).mma_major_mode()
        self.c_layout = utils.LayoutEnum.from_tensor(output)

        if cutlass.const_expr(dy_t.element_type != z_t.element_type):
            raise TypeError(f"input types must match: {dy_t.element_type} != {z_t.element_type}")

        self._setup_attributes()
        tiled_mma = self._create_tiled_mma()
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn,
            tiled_mma.thr_id,
        )
        a_smem_layout = cute.slice_(
            self.a_smem_layout_staged,
            (None, None, None, 0),
        )
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            dy_t,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn,
            tiled_mma.thr_id,
        )
        b_smem_layout = cute.slice_(
            self.b_smem_layout_staged,
            (None, None, None, 0),
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            z_t,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        a_copy_size = cute.size_in_bytes(self.ab_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.ab_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        epi_smem_layout = cute.slice_(
            self.c_smem_layout_staged,
            (None, None, 0),
        )
        tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
            cpasync.CopyReduceBulkTensorTileS2GOp(cute.ReductionKind.ADD),
            output,
            epi_smem_layout,
            self.epi_tile,
        )

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            expert_k_starts,
            expert_k_ends,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.c_smem_layout_staged,
            self.epi_tile,
        ).launch(
            grid=(self.num_pairs * 2, 1, 1),
            block=(self.threads_per_cta, 1, 1),
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        expert_k_starts: cute.Tensor,
        expert_k_ends: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        c_smem_layout_staged,
        epi_tile: cute.Tile,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_c)

        bidx, _, _ = cute.arch.block_idx()
        pair_rank = bidx % 2
        pair_idx = bidx // 2
        is_leader_cta = pair_rank == 0
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank)
        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        ab_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producers = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            num_tma_producers,
        )
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_producer_group,
            consumer_group=ab_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        acc_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumers = len(self.epilogue_warp_id) * (2 if use_2cta_instrs else 1)
        acc_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            num_acc_consumers,
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_producer_group,
            consumer_group=acc_consumer_group,
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
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        pipeline_init_arrive(
            cluster_shape_mn=cluster_layout_vmnk,
            is_relaxed=True,
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
            element_type=self.c_dtype,
            layout=c_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=c_smem_layout_staged.inner,
        )

        a_mcast_mask = None
        b_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk,
                cluster_coord_vmnk,
                mcast_mode=2,
            )
            b_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk,
                cluster_coord_vmnk,
                mcast_mode=1,
            )

        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.mma_tiler, (None, 0, None)),
            (None, None, None),
        )
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.mma_tiler, (0, None, None)),
            (None, None, None),
        )
        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.mma_tiler, (None, None, 0)),
            (None, None, None),
        )

        thr_mma = tiled_mma.get_slice(pair_rank)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(gC_mnl)

        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        if warp_idx == self.tma_warp_id:
            self.tma_mainloop(
                pair_idx,
                ab_producer,
                tma_atom_a,
                tAgA,
                tAsA,
                a_mcast_mask,
                tma_atom_b,
                tBgB,
                tBsB,
                b_mcast_mask,
                expert_k_starts,
                expert_k_ends,
            )

        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            self.mma_mainloop(
                pair_idx,
                is_leader_cta,
                tiled_mma,
                tCrA,
                tCrB,
                tCtAcc_base,
                ab_consumer,
                acc_pipeline,
                expert_k_starts,
                expert_k_ends,
            )

        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            self.epilogue_mainloop(
                pair_idx,
                tidx,
                warp_idx,
                tma_atom_c,
                tCtAcc_base,
                sC,
                tCgC,
                epi_tile,
                acc_pipeline,
                expert_k_starts,
                expert_k_ends,
            )
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

    @cute.jit
    def cell_coordinates(
        self,
        cell_idx: cutlass.Int32,
        expert_k_starts: cute.Tensor,
        expert_k_ends: cute.Tensor,
    ):
        k_slice = cell_idx % self.k_split
        cell_om = cell_idx // self.k_split
        chunk_idx = cell_om // self.outer_split
        split_lane = cell_om - chunk_idx * self.outer_split
        expert = chunk_idx // self.num_n_tiles
        n_tile = chunk_idx - expert * self.num_n_tiles

        walk_begin = split_lane * self.num_m_tiles // self.outer_split
        walk_end = (split_lane + 1) * self.num_m_tiles // self.outer_split
        kb_lo = cutlass.Int32(expert_k_starts[expert])
        kb_hi = cutlass.Int32(expert_k_ends[expert])
        k_total = kb_hi - kb_lo
        k_per_split = (k_total + self.k_split - 1) // self.k_split
        kb_lo = kb_lo + k_slice * k_per_split
        kb_hi = min(kb_lo + k_per_split, kb_hi)
        return expert, n_tile, walk_begin, walk_end, kb_lo, kb_hi

    @cute.jit
    def tma_mainloop(
        self,
        pair_idx,
        ab_producer,
        tma_atom_a,
        tAgA,
        tAsA,
        a_mcast_mask,
        tma_atom_b,
        tBgB,
        tBsB,
        b_mcast_mask,
        expert_k_starts,
        expert_k_ends,
    ):
        for cell_idx in cutlass.range(
            pair_idx,
            self.total_cells,
            self.num_pairs,
            unroll=1,
        ):
            (
                _,
                n_tile,
                walk_begin,
                walk_end,
                kb_lo,
                kb_hi,
            ) = self.cell_coordinates(
                cell_idx,
                expert_k_starts,
                expert_k_ends,
            )
            if kb_hi > kb_lo:
                for m_tile in cutlass.range(
                    walk_begin,
                    walk_end,
                    1,
                    unroll=1,
                ):
                    tAgA_slice = tAgA[(None, m_tile, None, 0)]
                    tBgB_slice = tBgB[(None, n_tile, None, 0)]
                    ab_producer.reset()
                    peek_empty = ab_producer.try_acquire()
                    for kb in cutlass.range(kb_lo, kb_hi, 1, unroll=1):
                        handle = ab_producer.acquire_and_advance(peek_empty)
                        cute.copy(
                            tma_atom_a,
                            tAgA_slice[(None, kb)],
                            tAsA[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            mcast_mask=a_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB_slice[(None, kb)],
                            tBsB[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            mcast_mask=b_mcast_mask,
                        )
                        peek_empty = cutlass.Boolean(1)
                        if kb + 1 < kb_hi:
                            peek_empty = ab_producer.try_acquire()
        ab_producer.tail()

    @cute.jit
    def mma_mainloop(
        self,
        pair_idx,
        is_leader_cta,
        tiled_mma,
        tCrA,
        tCrB,
        tCtAcc_base,
        ab_consumer,
        acc_pipeline,
        expert_k_starts,
        expert_k_ends,
    ):
        acc_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer,
            self.num_acc_stage,
        )
        for cell_idx in cutlass.range(
            pair_idx,
            self.total_cells,
            self.num_pairs,
            unroll=1,
        ):
            (
                _,
                _,
                walk_begin,
                walk_end,
                kb_lo,
                kb_hi,
            ) = self.cell_coordinates(
                cell_idx,
                expert_k_starts,
                expert_k_ends,
            )
            if kb_hi > kb_lo:
                for _ in cutlass.range(
                    walk_begin,
                    walk_end,
                    1,
                    unroll=1,
                ):
                    if is_leader_cta:
                        acc_pipeline.producer_acquire(acc_state)
                    tCtAcc = tCtAcc_base[(None, None, None, acc_state.index)]

                    ab_consumer.reset()
                    peek_full = cutlass.Boolean(1)
                    if is_leader_cta:
                        peek_full = ab_consumer.try_wait()
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                    for kb in cutlass.range(kb_lo, kb_hi, 1, unroll=1):
                        if is_leader_cta:
                            handle = ab_consumer.wait_and_advance(peek_full)
                            num_kblocks = cute.size(tCrA, mode=[2])
                            for kblk_idx in cutlass.range(
                                num_kblocks,
                                unroll_full=True,
                            ):
                                coord = (
                                    None,
                                    None,
                                    kblk_idx,
                                    handle.index,
                                )
                                cute.gemm(
                                    tiled_mma,
                                    tCtAcc,
                                    tCrA[coord],
                                    tCrB[coord],
                                    tCtAcc,
                                )
                                tiled_mma.set(
                                    tcgen05.Field.ACCUMULATE,
                                    True,
                                )
                            handle.release()
                            peek_full = cutlass.Boolean(1)
                            if kb + 1 < kb_hi:
                                peek_full = ab_consumer.try_wait()
                    if is_leader_cta:
                        acc_pipeline.producer_commit(acc_state)
                    acc_state.advance()
        acc_pipeline.producer_tail(acc_state)

    @cute.jit
    def epilogue_mainloop(
        self,
        pair_idx,
        tidx,
        warp_idx,
        tma_atom_c,
        tCtAcc_base,
        sC,
        tCgC,
        epi_tile,
        acc_pipeline,
        expert_k_starts,
        expert_k_ends,
    ):
        acc_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer,
            self.num_acc_stage,
        )
        c_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 128)
        c_pipeline = pipeline.PipelineTmaStore.create(
            num_stages=self.num_c_stage // self.num_epi_wg,
            producer_group=c_group,
        )
        tiles_executed = cutlass.Int32(0)

        for cell_idx in cutlass.range(
            pair_idx,
            self.total_cells,
            self.num_pairs,
            unroll=1,
        ):
            (
                expert,
                n_tile,
                walk_begin,
                walk_end,
                kb_lo,
                kb_hi,
            ) = self.cell_coordinates(
                cell_idx,
                expert_k_starts,
                expert_k_ends,
            )
            if kb_hi > kb_lo:
                for m_tile in cutlass.range(
                    walk_begin,
                    walk_end,
                    1,
                    unroll=1,
                ):
                    acc_state = self.epilogue(
                        tidx,
                        warp_idx,
                        tma_atom_c,
                        tCtAcc_base,
                        sC,
                        tCgC,
                        epi_tile,
                        tiles_executed,
                        (m_tile, n_tile, expert),
                        acc_state,
                        acc_pipeline,
                        c_pipeline,
                    )
                    tiles_executed += 1
        c_pipeline.producer_tail()

    @cute.jit
    def epilogue(
        self,
        tidx,
        warp_idx,
        tma_atom_c,
        tCtAcc_base,
        sC,
        tCgC_base,
        epi_tile,
        tiles_executed,
        tile_coord,
        acc_state,
        acc_pipeline,
        c_pipeline,
    ):
        epi_wg = warp_idx // 4
        epi_tidx = tidx % 128

        tCgC = gemm_sm100.transform_partitioned_tensor_layout(tCgC_base)
        tCtAcc = gemm_sm100.transform_partitioned_tensor_layout(tCtAcc_base)
        tiled_t2r, tTR_tAcc_base, tTR_rAcc = gemm_sm100.epilogue_tmem_copy_and_partition(
            self,
            epi_tidx,
            tCtAcc,
            tCgC,
            epi_tile,
            True,
        )
        tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, self.c_dtype)
        tiled_r2s, tRS_rC, tRS_sC = gemm_sm100.epilogue_smem_copy_and_partition(
            self,
            tiled_t2r,
            tTR_rC,
            epi_tidx,
            sC,
        )

        tCgC_epi = cute.flat_divide(tCgC, epi_tile)
        bSG_sC, bSG_gC_partitioned = cpasync.tma_partition(
            tma_atom_c,
            0,
            cute.make_layout(1),
            cute.group_modes(sC, 0, 2),
            cute.group_modes(tCgC_epi, 0, 2),
        )
        bSG_gC = bSG_gC_partitioned[(None, None, None, *tile_coord)]

        barrier = pipeline.NamedBarrier(
            barrier_id=self.epilog_sync_bar_id + epi_wg,
            num_threads=128,
        )
        tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_state.index)]
        acc_pipeline.consumer_wait(acc_state)

        tTR_tAcc = cute.group_modes(
            tTR_tAcc,
            3,
            cute.rank(tTR_tAcc),
        )
        bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

        subtile_count = cute.size(tTR_tAcc.shape, mode=[3])
        subtiles_per_wg = subtile_count // self.num_epi_wg
        stages_per_wg = self.num_c_stage // self.num_epi_wg
        wg_subtile_base = epi_wg * subtiles_per_wg
        wg_stage_base = epi_wg * stages_per_wg
        previous_subtiles = tiles_executed * subtiles_per_wg

        for local_idx in range(subtiles_per_wg):
            subtile_idx = wg_subtile_base + local_idx
            cute.copy(
                tiled_t2r,
                tTR_tAcc[(None, None, None, subtile_idx)],
                tTR_rAcc,
            )
            acc_vec = tiled_r2s.retile(tTR_rAcc).load()
            tRS_rC.store(acc_vec.to(self.c_dtype))

            c_buffer = wg_stage_base + (previous_subtiles + local_idx) % stages_per_wg
            cute.copy(
                tiled_r2s,
                tRS_rC,
                tRS_sC[(None, None, None, c_buffer)],
            )
            cute.arch.fence_proxy("async.shared", space="cta")
            barrier.arrive_and_wait()

            if warp_idx % 4 == 0:
                cute.copy(
                    tma_atom_c,
                    bSG_sC[(None, c_buffer)],
                    bSG_gC[(None, subtile_idx)],
                )
                c_pipeline.producer_commit()
                c_pipeline.producer_acquire()
            barrier.arrive_and_wait()

        barrier.arrive_and_wait()
        with cute.arch.elect_one():
            acc_pipeline.consumer_release(acc_state)
        acc_state.advance()
        return acc_state

    @staticmethod
    def _compute_stages(
        tiled_mma,
        mma_tiler_mnk,
        ab_dtype,
        epi_tile,
        c_dtype,
        c_layout,
        smem_capacity,
        occupancy,
    ) -> Tuple[int, int]:
        num_c_stage = 2
        a_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            ab_dtype,
            1,
        )
        b_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            ab_dtype,
            1,
        )
        c_one = sm100_utils.make_smem_layout_epi(
            c_dtype,
            c_layout,
            epi_tile,
            1,
        )
        ab_bytes = cute.size_in_bytes(
            ab_dtype,
            a_one,
        ) + cute.size_in_bytes(ab_dtype, b_one)
        c_stage_bytes = cute.size_in_bytes(c_dtype, c_one)
        fixed_bytes = 1024
        c_bytes = c_stage_bytes * num_c_stage
        num_ab_stage = (smem_capacity // occupancy - fixed_bytes - c_bytes) // ab_bytes
        num_c_stage += (smem_capacity - occupancy * ab_bytes * num_ab_stage - occupancy * (fixed_bytes + c_bytes)) // (
            occupancy * c_stage_bytes
        )
        return num_ab_stage, num_c_stage


def _validate_inputs(dy, z, starts, ends):
    if dy.ndim != 2 or z.ndim != 2:
        raise ValueError("dy and z must be rank-2")
    if dy.shape[0] != z.shape[0]:
        raise ValueError("dy and z token dimensions must match")
    if dy.dtype != torch.bfloat16 or z.dtype != torch.bfloat16:
        raise TypeError("dy and z must be BF16")
    if dy.device != z.device or not dy.is_cuda:
        raise ValueError("dy and z must share a CUDA device")
    if dy.shape[0] % TILE_K or dy.shape[1] % TILE_M or z.shape[1] % TILE_N:
        raise ValueError(f"shapes must divide tile {(TILE_K, TILE_M, TILE_N)}")
    if starts.shape != ends.shape:
        raise ValueError("expert range tensors must have identical shapes")
    if starts.dtype != torch.int32 or ends.dtype != torch.int32:
        raise TypeError("expert ranges must use int32")


def prepare(dy, z, expert_k_starts, expert_k_ends):
    _validate_inputs(dy, z, expert_k_starts, expert_k_ends)
    return {
        "dy": dy,
        "z": z,
        "expert_k_starts": expert_k_starts,
        "expert_k_ends": expert_k_ends,
        "output": torch.empty(
            (
                expert_k_starts.numel(),
                dy.shape[1],
                z.shape[1],
            ),
            dtype=torch.bfloat16,
            device=dy.device,
        ),
    }


def launch(state, outer_split, k_split=1):
    dy = state["dy"]
    z = state["z"]
    starts = state["expert_k_starts"]
    ends = state["expert_k_ends"]
    output = state["output"]

    num_m_tiles = dy.shape[1] // TILE_M
    if outer_split < 1 or num_m_tiles % outer_split:
        raise ValueError(f"outer_split={outer_split} must divide {num_m_tiles}")
    if k_split < 1:
        raise ValueError("k_split must be positive")

    num_experts = starts.numel()
    num_n_tiles = z.shape[1] // TILE_N
    total_cells = num_experts * num_n_tiles * outer_split * k_split
    max_clusters = _max_active_clusters(2)
    num_pairs = max(1, min(max_clusters, total_cells))

    dy_ct = _to_cute_tensor(_as_mkl(dy.T), leading_dim=0)
    z_ct = _to_cute_tensor(_as_mkl(z.T), leading_dim=0)
    output_ct = _to_cute_tensor(_as_mnl(output), leading_dim=1)
    starts_ct = _to_cute_tensor(starts, leading_dim=0)
    ends_ct = _to_cute_tensor(ends, leading_dim=0)
    stream = _cute_stream()

    key = (
        dy.dtype,
        tuple(dy.shape),
        tuple(z.shape),
        num_experts,
        outer_split,
        k_split,
        num_pairs,
    )
    compiled = _compile_cache.get(key)
    if compiled is None:
        kernel = Mlp3TwoSmPersistentKernel(
            acc_dtype=cutlass.Float32,
            outer_split=outer_split,
            k_split=k_split,
            num_experts=num_experts,
            num_m_tiles=num_m_tiles,
            num_n_tiles=num_n_tiles,
            num_pairs=num_pairs,
        )
        compiled = cute.compile(
            kernel,
            dy_ct,
            z_ct,
            output_ct,
            starts_ct,
            ends_ct,
            stream,
        )
        _compile_cache[key] = compiled
        _compile_metadata[key] = {
            "mainloop_stages": kernel.num_ab_stage,
            "epilogue_stages": kernel.num_c_stage,
            "accumulator_stages": kernel.num_acc_stage,
            "epilogue_warpgroups": kernel.num_epi_wg,
            "tmem_columns": kernel.num_tmem_alloc_cols,
            "mainloop_smem_bytes": kernel.mainloop_smem_bytes,
            "epilogue_smem_bytes": kernel.epilogue_smem_bytes,
            "smem_budget_bytes": kernel.smem_budget_bytes,
        }
        compiled(
            dy_ct,
            z_ct,
            output_ct,
            starts_ct,
            ends_ct,
            stream,
        )
        return output

    compiled(
        dy_ct,
        z_ct,
        output_ct,
        starts_ct,
        ends_ct,
        stream,
    )
    return output


def compile_metadata():
    """Return metadata for the most recently compiled kernel specialization."""
    if not _compile_metadata:
        return None
    return next(reversed(_compile_metadata.values()))
