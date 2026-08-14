#pragma once

// Experimental SM100 paired-CTA MLP1:
//   Z = SiLU(X @ B^T) * (X @ C^T)
//
// Two CTAs issue one joined 128x256 UMMA. Each CTA owns a 64x256 accumulator
// fragment. SM100 interleaves that fragment over all 128 TMEM datapaths, so one
// logical accumulator occupies 128 physical columns. U/V double buffering
// therefore uses 2 stages * 2 accumulators * 128 columns = 512 columns.

#include <cute/tensor.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/copy_traits_sm100.hpp>
#include <cute/atom/copy_traits_sm100_tma.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/pipeline/sm100_pipeline.hpp>

#include "math.cuh"
#include "tmem_load_op.cuh"

namespace liger {

using namespace cute;

template <
	typename Element_,
	int TileM_ = 128,
	int TileN_ = 256,
	int TileK_ = 64,
	int Stages_ = 4,
	int EpiChunkN_ = 64,
	int AccStages_ = 2>
struct Mlp1Traits2Sm {
	using Element = Element_;
	using ElementAccum = float;

	static constexpr int TileM = TileM_;
	static constexpr int TileN = TileN_;
	static constexpr int TileK = TileK_;
	static constexpr int Stages = Stages_;
	static constexpr int EpiChunkN = EpiChunkN_;
	static constexpr int AccStages = AccStages_;
	static constexpr int ClusterM = 2;
	static constexpr int CtaTileM = TileM / ClusterM;
	static constexpr int WgTileN = TileN / 2;
	static constexpr int NumEpiRounds = WgTileN / EpiChunkN;

	static_assert(TileM == 128, "MLP1 2SM experiment requires joined TileM=128");
	static_assert(TileN == 256, "MLP1 2SM experiment requires joined TileN=256");
	static_assert(TileK == 32 || TileK == 64, "MLP1 2SM supports TileK=32 or 64");
	static_assert(Stages >= 2 && Stages <= 8, "MLP1 2SM supports two to eight mainloop stages");
	static_assert(CtaTileM == 64, "each peer CTA must own 64 M rows");
	static_assert(WgTileN % EpiChunkN == 0, "EpiChunkN must divide each warpgroup's N half");
	static_assert(AccStages == 2, "the experiment keeps accumulator double buffering");

	using TileShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
	using ClusterShape = Shape<Int<ClusterM>, _1, _1>;
	using AtomThrShape = Shape<_2, _1, _1>;

	using TiledMma2Sm = decltype(make_tiled_mma(
		SM100_MMA_F16BF16_2x1SM_SS<
			Element, Element, ElementAccum, TileM, TileN,
			UMMA::Major::K, UMMA::Major::K>{}));
	static_assert(size(typename TiledMma2Sm::AtomThrID{}) == 2,
		"MLP1 2SM requires a two-CTA MMA atom");

	using MmaShapeX_MK = decltype(partition_shape_A(
		TiledMma2Sm{}, make_shape(Int<TileM>{}, Int<TileK>{})));
	using MmaShapeW_NK = decltype(partition_shape_B(
		TiledMma2Sm{}, make_shape(Int<TileN>{}, Int<TileK>{})));

	using SmemLayoutAtom = UMMA::Layout_K_SW128_Atom<Element>;
	using SmemLayoutX = decltype(UMMA::tile_to_mma_shape(
		SmemLayoutAtom{}, append(MmaShapeX_MK{}, Int<Stages>{}),
		Step<_2, _1, _3>{}));
	using SmemLayoutW = decltype(UMMA::tile_to_mma_shape(
		SmemLayoutAtom{}, append(MmaShapeW_NK{}, Int<Stages>{}),
		Step<_2, _1, _3>{}));
	using SmemLayoutX_1 = decltype(SmemLayoutX{}(_, _, _, Int<0>{}));
	using SmemLayoutW_1 = decltype(SmemLayoutW{}(_, _, _, Int<0>{}));

	static constexpr int TmaTransBytesX =
		static_cast<int>(cosize_v<SmemLayoutX_1> * sizeof(Element));
	static constexpr int TmaTransBytesW =
		static_cast<int>(cosize_v<SmemLayoutW_1> * sizeof(Element));
	// A 2SM TMA transaction contributes one peer-local slice from each CTA.
	static constexpr int TmaTransBytes =
		2 * (TmaTransBytesX + 2 * TmaTransBytesW);

	using MainloopPipeline = cutlass::PipelineTmaUmmaAsync<
		Stages, ClusterShape, AtomThrShape>;
	using PipelineState = typename MainloopPipeline::PipelineState;
	using AccumulatorPipeline = cutlass::PipelineUmmaAsync<
		AccStages, AtomThrShape>;

	// For joined M=128, each CTA's M=64 fragment uses the 2x2 interleaved
	// TMEM layout: logical N=256 occupies N/2=128 physical columns.
	static constexpr int TmemColumnsPerAccumulator = TileN / 2;
	static constexpr int TmemColumns =
		AccStages * 2 * TmemColumnsPerAccumulator;
	static_assert(TmemColumns == 512, "MLP1 2SM must exactly fit the TMEM column budget");

	static constexpr int WarpSize = 32;
	static constexpr int WarpGroupSize = 128;
	static constexpr int NumConsumers = 2;
	static constexpr int ConsumerThreads = NumConsumers * WarpGroupSize;
	static constexpr int NumThreads = 384;

	using SmemLayoutStoreSlot = Layout<
		Shape<Int<CtaTileM>, Int<EpiChunkN>>,
		Stride<Int<EpiChunkN>, _1>>;
};

template <typename Traits>
struct Mlp1Fused2SmSmem {
	using Element = typename Traits::Element;

	static constexpr int smem_X_size = cosize_v<typename Traits::SmemLayoutX>;
	static constexpr int smem_W_size = cosize_v<typename Traits::SmemLayoutW>;
	static constexpr int smem_store_size =
		cosize_v<typename Traits::SmemLayoutStoreSlot>;

	alignas(128) Element smem_X[smem_X_size];
	alignas(128) Element smem_W1[smem_W_size];
	alignas(128) Element smem_W2[smem_W_size];
	alignas(128) Element store_buf[2 * smem_store_size];
	alignas(16) typename Traits::MainloopPipeline::SharedStorage pipe_storage;
	alignas(16) uint32_t tmem_base;
	alignas(16) typename Traits::AccumulatorPipeline::SharedStorage acc_pipe;

	CUTE_DEVICE Element* X_data() { return &smem_X[0]; }
	CUTE_DEVICE Element* W1_data() { return &smem_W1[0]; }
	CUTE_DEVICE Element* W2_data() { return &smem_W2[0]; }
};

template <typename Traits>
__device__ __forceinline__ typename Traits::MainloopPipeline
mlp1_make_pipe_umma_2sm(
		typename Traits::MainloopPipeline::SharedStorage& storage) {
	using Pipeline = typename Traits::MainloopPipeline;
	using Category = typename Pipeline::ThreadCategory;

	int warp_id = threadIdx.x / Traits::WarpSize;
	bool is_producer = warp_id == 0;
	bool is_consumer = warp_id >= 3 && warp_id <= 11;

	typename Pipeline::Params params;
	params.transaction_bytes = Traits::TmaTransBytes;
	params.num_producers = 1;
	params.num_consumers = 1;
	params.initializing_warp = 0;
	if (is_producer) {
		params.role = Category::Producer;
		params.is_leader =
			threadIdx.x == 0 && cute::block_rank_in_cluster() == 0;
	} else if (is_consumer) {
		params.role = Category::Consumer;
	} else {
		params.role = Category::NonParticipant;
	}
	return Pipeline(
		storage, params, typename Traits::ClusterShape{},
		cute::true_type{}, cute::true_type{});
}

template <
	typename Traits,
	typename TmaLoadX,
	typename TmaLoadW,
	typename TmaStoreZ>
__device__ __forceinline__ void mlp1_fused_2sm(
		Mlp1Fused2SmSmem<Traits>& smem,
		TmaLoadX const& tma_load_x,
		TmaLoadW const& tma_load_b,
		TmaLoadW const& tma_load_c,
		TmaStoreZ const& tma_store_z,
		const int* expert_ids,
		int num_tokens,
		int hidden_dim,
		int total_n_rows,
		int num_m_tiles,
		int num_n_tiles) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
	using Element = typename Traits::Element;
	using Pipeline = typename Traits::MainloopPipeline;
	using PipeState = typename Traits::PipelineState;
	using AccPipe = typename Traits::AccumulatorPipeline;

	constexpr int TileM = Traits::TileM;
	constexpr int TileN = Traits::TileN;
	constexpr int CtaTileM = Traits::CtaTileM;
	constexpr int EpiChunkN = Traits::EpiChunkN;
	constexpr int NumEpiRounds = Traits::NumEpiRounds;
	constexpr int kMmaEpiThreads =
		Traits::ConsumerThreads + Traits::WarpSize;

	int warp_id = threadIdx.x / Traits::WarpSize;
	int pair_rank = static_cast<int>(cute::block_rank_in_cluster());
	bool is_leader_cta = pair_rank == 0;
	bool is_producer = warp_id == 0;
	bool is_mma_warp = warp_id == 3;
	bool is_epilogue = warp_id >= 4 && warp_id <= 11;
	int tid_in_epi = threadIdx.x - Traits::WarpGroupSize;
	int wg = is_epilogue ? tid_in_epi / Traits::WarpGroupSize : 0;
	int tid_in_wg = is_epilogue ? tid_in_epi % Traits::WarpGroupSize : 0;
	int wg_barrier_id = 1 + wg;
	bool is_wg_leader = is_epilogue && tid_in_wg == 0;

	cute::prefetch_tma_descriptor(tma_load_x.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_b.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_load_c.get_tma_descriptor());
	cute::prefetch_tma_descriptor(tma_store_z.get_tma_descriptor());

	Pipeline pipe = mlp1_make_pipe_umma_2sm<Traits>(smem.pipe_storage);
	PipeState prod_state = cutlass::make_producer_start_state<Pipeline>();
	PipeState cons_state;

	typename Traits::TiledMma2Sm tiled_mma;
	auto cta_mma = tiled_mma.get_slice(pair_rank);
	auto pair_layout_vmnk = tiled_divide(
		make_layout(typename Traits::ClusterShape{}),
		make_tile(typename Traits::TiledMma2Sm::AtomThrID{}));
	auto pair_coord_vmnk = pair_layout_vmnk.get_flat_coord(pair_rank);

	auto sX = make_tensor(
		make_smem_ptr(smem.X_data()), typename Traits::SmemLayoutX{});
	auto sW1 = make_tensor(
		make_smem_ptr(smem.W1_data()), typename Traits::SmemLayoutW{});
	auto sW2 = make_tensor(
		make_smem_ptr(smem.W2_data()), typename Traits::SmemLayoutW{});

	auto mX = tma_load_x.get_tma_tensor(make_shape(
		static_cast<int64_t>(num_tokens),
		static_cast<int64_t>(hidden_dim)));
	auto mB = tma_load_b.get_tma_tensor(make_shape(
		static_cast<int64_t>(total_n_rows),
		static_cast<int64_t>(hidden_dim)));
	auto mC = tma_load_c.get_tma_tensor(make_shape(
		static_cast<int64_t>(total_n_rows),
		static_cast<int64_t>(hidden_dim)));
	auto mZ = tma_store_z.get_tma_tensor(make_shape(
		static_cast<int64_t>(num_m_tiles) * TileM,
		static_cast<int64_t>(num_n_tiles) * TileN));

	uint16_t mcast_mask_x =
		create_tma_multicast_mask<2>(pair_layout_vmnk, pair_coord_vmnk);
	uint16_t mcast_mask_w =
		create_tma_multicast_mask<1>(pair_layout_vmnk, pair_coord_vmnk);

	auto tCrX = cta_mma.make_fragment_A(sX);
	auto tCrW1 = cta_mma.make_fragment_B(sW1);
	auto tCrW2 = cta_mma.make_fragment_B(sW2);

	auto cAccFull = make_identity_tensor(
		make_shape(Int<TileM>{}, Int<TileN>{}));
	auto tCgC = cta_mma.partition_C(cAccFull);
	auto tCtAccU = cta_mma.make_fragment_C(tCgC);
	auto tCtAccV = cta_mma.make_fragment_C(tCgC);

	typename AccPipe::Params acc_params;
	if (is_mma_warp && is_leader_cta) {
		acc_params.role = AccPipe::ThreadCategory::Producer;
	} else if (warp_id >= 3 && warp_id <= 11) {
		acc_params.role = AccPipe::ThreadCategory::Consumer;
	} else {
		acc_params.role = AccPipe::ThreadCategory::NonParticipant;
	}
	acc_params.producer_arv_count = 1;
	acc_params.consumer_arv_count = Traits::ClusterM;
	acc_params.initializing_warp = 4;
	AccPipe acc_pipe(
		smem.acc_pipe, acc_params, typename Traits::ClusterShape{});
	auto acc_prod_state = cutlass::make_producer_start_state<AccPipe>();
	typename AccPipe::PipelineState acc_cons_state;

	cute::TMEM::Allocator2Sm tmem_allocator;
	cute::cluster_sync();
	if (warp_id == 4) {
		tmem_allocator.allocate(Traits::TmemColumns, &smem.tmem_base);
		__syncwarp();
	}
	__syncthreads();
	cute::cluster_sync();
	if (warp_id >= 3 && warp_id <= 11)
		cutlass::arch::NamedBarrier::sync(kMmaEpiThreads, 3);
	uint32_t tmem_base = smem.tmem_base;

	constexpr int store_slot_elems = CtaTileM * EpiChunkN;
	Element* store_ptr = smem.store_buf + wg * store_slot_elems;
	auto sStore = make_tensor(
		make_smem_ptr(store_ptr),
		typename Traits::SmemLayoutStoreSlot{});
	auto cta_tma_z = tma_store_z.get_slice(Int<0>{});

	auto epi_tile = make_tile(Int<CtaTileM>{}, Int<EpiChunkN>{});
	tCtAccU.data() = tmem_base;
	tCtAccV.data() =
		tmem_base + uint32_t(Traits::TmemColumnsPerAccumulator);
	auto accU_mn = tCtAccU(make_coord(_, _), _0{}, _0{});
	auto tAccU_epi = flat_divide(accU_mn, epi_tile);
	auto t2r = make_tmem_copy(
		TmemLoadOp<EpiChunkN>{},
		tAccU_epi(_, _, _0{}, _0{}));
	auto thr_t2r = t2r.get_slice(tid_in_wg);
	auto cChunk = make_identity_tensor(
		make_shape(Int<CtaTileM>{}, Int<EpiChunkN>{}));
	auto tTR_cChunk = thr_t2r.partition_D(cChunk);
	auto tTR_rU = make_tensor<float>(shape(tTR_cChunk));
	auto tTR_rV = make_tensor<float>(shape(tTR_cChunk));

	Layout tmem_warp_layout =
		typename decltype(make_tmem_warp_partitioner(
			tAccU_epi(_, _, _0{}, _0{})))::TiledLayout_TV{};
	constexpr bool predicate_tmem_load =
		size(tmem_warp_layout) != cosize(tmem_warp_layout);

	int num_k_tiles = hidden_dim / Traits::TileK;
	int total_cells = num_m_tiles * num_n_tiles;
	int cell_start = static_cast<int>(blockIdx.x) / Traits::ClusterM;
	int cell_stride = static_cast<int>(gridDim.x) / Traits::ClusterM;
	bool store_in_flight = false;

	for (int cell = cell_start; cell < total_cells; cell += cell_stride) {
		int m = cell / num_n_tiles;
		int n = cell - m * num_n_tiles;
		int expert = expert_ids[m];

		if (is_producer) {
			auto coord = make_coord(m, n, _);
			auto gX = local_tile(
				mX, typename Traits::TileShape{}, coord,
				Step<_1, X, _1>{});
			auto gB = local_tile(
				mB, typename Traits::TileShape{},
				make_coord(_, expert * num_n_tiles + n, _),
				Step<X, _1, _1>{});
			auto gC = local_tile(
				mC, typename Traits::TileShape{},
				make_coord(_, expert * num_n_tiles + n, _),
				Step<X, _1, _1>{});
			auto tCgX = cta_mma.partition_A(gX);
			auto tCgB = cta_mma.partition_B(gB);
			auto tCgW2 = cta_mma.partition_B(gC);

			auto [tXgX, tXsX] = tma_partition(
				tma_load_x,
				get<2>(pair_coord_vmnk),
				make_layout(size<2>(pair_layout_vmnk)),
				group_modes<0, 3>(sX),
				group_modes<0, 3>(tCgX));
			auto [tBgB, tBsW1] = tma_partition(
				tma_load_b,
				get<1>(pair_coord_vmnk),
				make_layout(size<1>(pair_layout_vmnk)),
				group_modes<0, 3>(sW1),
				group_modes<0, 3>(tCgB));
			auto [tCgC, tCsW2] = tma_partition(
				tma_load_c,
				get<1>(pair_coord_vmnk),
				make_layout(size<1>(pair_layout_vmnk)),
				group_modes<0, 3>(sW2),
				group_modes<0, 3>(tCgW2));

			for (int k = 0; k < num_k_tiles; ++k) {
				pipe.producer_acquire(prod_state);
				if (cute::elect_one_sync()) {
					auto* barrier = pipe.producer_get_barrier(prod_state);
					copy(
						tma_load_x.with(*barrier, mcast_mask_x),
						tXgX(_, k),
						tXsX(_, prod_state.index()));
					copy(
						tma_load_b.with(*barrier, mcast_mask_w),
						tBgB(_, k),
						tBsW1(_, prod_state.index()));
					copy(
						tma_load_c.with(*barrier, mcast_mask_w),
						tCgC(_, k),
						tCsW2(_, prod_state.index()));
				}
				++prod_state;
			}
		}

		if (is_mma_warp && is_leader_cta) {
			acc_pipe.producer_acquire(acc_prod_state);
			int acc_stage = acc_prod_state.index();
			uint32_t stage_base = tmem_base +
				uint32_t(acc_stage * 2 * Traits::TmemColumnsPerAccumulator);
			tCtAccU.data() = stage_base;
			tCtAccV.data() =
				stage_base + uint32_t(Traits::TmemColumnsPerAccumulator);

			for (int k = 0; k < num_k_tiles; ++k) {
				pipe.consumer_wait(cons_state);
				CUTE_UNROLL
				for (int kb = 0; kb < size<2>(tCrX); ++kb) {
					tiled_mma.accumulate_ = (k == 0 && kb == 0)
						? UMMA::ScaleOut::Zero
						: UMMA::ScaleOut::One;
					gemm(
						tiled_mma,
						tCrX(_, _, kb, cons_state.index()),
						tCrW1(_, _, kb, cons_state.index()),
						tCtAccU);
					gemm(
						tiled_mma,
						tCrX(_, _, kb, cons_state.index()),
						tCrW2(_, _, kb, cons_state.index()),
						tCtAccV);
				}
				pipe.consumer_release(cons_state);
				++cons_state;
			}
			acc_pipe.producer_commit(acc_prod_state);
			++acc_prod_state;
		}

		if (is_epilogue) {
			acc_pipe.consumer_wait(acc_cons_state);
			int acc_stage = acc_cons_state.index();
			uint32_t stage_base = tmem_base +
				uint32_t(acc_stage * 2 * Traits::TmemColumnsPerAccumulator);
			tCtAccU.data() = stage_base;
			tCtAccV.data() =
				stage_base + uint32_t(Traits::TmemColumnsPerAccumulator);
			auto accU_mn_stage =
				tCtAccU(make_coord(_, _), _0{}, _0{});
			auto accV_mn_stage =
				tCtAccV(make_coord(_, _), _0{}, _0{});
			auto tAccU_epi_stage =
				flat_divide(accU_mn_stage, epi_tile);
			auto tAccV_epi_stage =
				flat_divide(accV_mn_stage, epi_tile);
			auto tTR_tAccU = thr_t2r.partition_S(tAccU_epi_stage);
			auto tTR_tAccV = thr_t2r.partition_S(tAccV_epi_stage);

			CUTE_UNROLL
			for (int round = 0; round < NumEpiRounds; ++round) {
				int chunk = wg * NumEpiRounds + round;
				auto tAccUChunk =
					tTR_tAccU(_, _, _, _0{}, chunk);
				auto tAccVChunk =
					tTR_tAccV(_, _, _, _0{}, chunk);
				bool issue_tmem_load = true;
				if constexpr (predicate_tmem_load) {
					int subpart =
						(tAccUChunk.data().dp_ / 32) % 4;
					issue_tmem_load =
						tid_in_wg / Traits::WarpSize == subpart;
				}
				if (issue_tmem_load) {
					copy(t2r, tAccUChunk, tTR_rU);
					copy(t2r, tAccVChunk, tTR_rV);
					CUTE_UNROLL
					for (int i = 0; i < size(tTR_rU); ++i)
						tTR_rU(i) =
							fast_silu(tTR_rU(i)) * tTR_rV(i);
				}

				if (store_in_flight)
					cute::tma_store_wait<0>();
				cutlass::arch::NamedBarrier::sync(
					Traits::WarpGroupSize, wg_barrier_id);
				if (issue_tmem_load) {
					CUTE_UNROLL
					for (int i = 0; i < size(tTR_rU); ++i) {
						int m_local = get<0>(tTR_cChunk(i));
						int n_local = get<1>(tTR_cChunk(i));
						sStore(m_local, n_local) =
							static_cast<Element>(tTR_rU(i));
					}
				}
				cutlass::arch::NamedBarrier::sync(
					Traits::WarpGroupSize, wg_barrier_id);

				if (is_wg_leader) {
					cute::tma_store_fence();
					int m_tile_idx = Traits::ClusterM * m + pair_rank;
					int n_tile_idx =
						n * (TileN / EpiChunkN) + chunk;
					auto gZ = local_tile(
						mZ,
						make_tile(
							Int<CtaTileM>{},
							Int<EpiChunkN>{}),
						make_coord(m_tile_idx, n_tile_idx));
					copy(
						tma_store_z,
						cta_tma_z.partition_S(sStore),
						cta_tma_z.partition_D(gZ));
					cute::tma_store_arrive();
				}
				store_in_flight = true;
			}

			cutlass::arch::NamedBarrier::sync(
				Traits::ConsumerThreads, 0);
			if (tid_in_epi == 0)
				acc_pipe.consumer_release(acc_cons_state);
			++acc_cons_state;
		}
	}

	if (is_producer) {
		pipe.producer_tail(prod_state);
	}
	if (is_epilogue && store_in_flight)
		cute::tma_store_wait<0>();

	if (warp_id >= 3 && warp_id <= 11)
		cutlass::arch::NamedBarrier::sync(kMmaEpiThreads, 3);
	__syncthreads();
	cute::cluster_sync();
	if (warp_id == 4) {
		tmem_allocator.release_allocation_lock();
		tmem_allocator.free(smem.tmem_base, Traits::TmemColumns);
	}
#else
	__trap();
#endif
}

}  // namespace liger
