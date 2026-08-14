#pragma once

// Experimental SM100 2-SM MLP3 path. Callers must launch with
// cudaLaunchKernelEx, clusterDim=(2,1,1), an even grid.x, and pair-aware
// SM100 2SM TMA load descriptors. The dA descriptor remains TMA REDUCE_ADD.

#include "mlp3.cuh"

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/copy_traits_sm100_tma.hpp>

namespace liger {

template <
	typename Element_,
	int TileM_ = 256,
	int TileN_ = 256,
	int TileK_ = 64,
	int Stages_ = 3,
	int EpiChunkN_ = 64,
	int AccStages_ = 2,
	int ClusterM_ = 2,
	bool CompactEpilogue_ = false,
	// Experiment (default off): release the accumulator TMEM stage
	// (acc_pipe.consumer_release) immediately after the final epilogue
	// round's synchronous TMEM->register copy instead of after all of that
	// round's register->SMEM fold + TMA_REDUCE_ADD store work. See the
	// EarlyTmemRelease branch in mlp3_consumer_2sm() for the correctness
	// argument. Compile-time gated so CompactEpilogue and non-compact
	// production behavior is byte-for-byte unchanged when false.
	bool EarlyTmemRelease_ = false,
	bool MulticastShared_ = false>
struct Mlp3Traits2Sm {
	using Element = Element_;
	using ElementAccum = float;

	static constexpr int TileM = TileM_;
	static constexpr int TileN = TileN_;
	static constexpr int TileK = TileK_;
	static constexpr int Stages = Stages_;
	static constexpr int EpiChunkN = EpiChunkN_;
	static constexpr int AccStages = AccStages_;
	static constexpr int ClusterM = ClusterM_;
	static constexpr int NumPairs = ClusterM / 2;
	static constexpr bool CompactEpilogue = CompactEpilogue_;
	static constexpr bool EarlyTmemRelease = EarlyTmemRelease_;
	static constexpr bool MulticastShared = MulticastShared_;
	static constexpr bool kIsMlp3TwoSm = true;
	// See Mlp3Traits::kIsMlp3Nsplit / mlp3_nsplit.cuh — false here since the
	// 2SM trait is never the experimental 1SM N-split path.
	static constexpr bool kIsMlp3Nsplit = false;

	static constexpr int CtaTileM = TileM / 2;
	static constexpr int AtomTileM = 64;
	static constexpr int WgTileN = TileN / 2;
	static constexpr int NumEpiRounds = WgTileN / EpiChunkN;
	static constexpr int kAtomsPerCta = CtaTileM / AtomTileM;

	static_assert(TileM == 256,
		"MLP3 2SM requires joined TileM=256; joined TileM=128 is invalid");
	static_assert(TileN == 128 || TileN == 256,
		"MLP3 2SM supports TileN=128 or TileN=256");
	static_assert(TileK == 32 || TileK == 64,
		"MLP3 2SM pilot supports TileK=32 or 64");
	static_assert(Stages >= 2 && Stages <= 12,
		"MLP3 2SM supports two to twelve mainloop stages");
	static_assert(CtaTileM == 128, "each peer CTA must own 128 M rows");
	static_assert(WgTileN % EpiChunkN == 0,
		"EpiChunkN must divide each warpgroup's N half");
	static_assert(kAtomsPerCta == 2,
		"each CTA must reduce-add two 64-row atoms");
	static_assert(AccStages >= 2, "MLP3 2SM requires accumulator double buffering");
	static_assert(AccStages * TileN <= 512,
		"MLP3 2SM accumulator stages must fit in 512 TMEM columns");
	static_assert(ClusterM == 2 || ClusterM == 4,
		"MLP3 2SM supports a one-pair or two-pair M cluster");
	static_assert(!MulticastShared || ClusterM == 4,
		"MLP3 shared-operand multicast requires two 2SM pairs");

	using TileShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;
	using ClusterShape = Shape<Int<ClusterM>, _1, _1>;
	using AtomThrShape = Shape<_2, _1, _1>;

	using TiledMma2Sm = decltype(make_tiled_mma(
		SM100_MMA_F16BF16_2x1SM_SS<
			Element, Element, ElementAccum, TileM, TileN,
			UMMA::Major::MN, UMMA::Major::MN>{}));
	static_assert(size(typename TiledMma2Sm::AtomThrID{}) == 2,
		"MLP3 2SM requires a two-CTA MMA atom");

	using ClusterLayoutVMNK = decltype(tiled_divide(
		make_layout(ClusterShape{}),
		make_tile(typename TiledMma2Sm::AtomThrID{})));

	using MmaShapeA_MK = decltype(partition_shape_A(
		TiledMma2Sm{}, make_shape(Int<TileM>{}, Int<TileK>{})));
	using MmaShapeB_NK = decltype(partition_shape_B(
		TiledMma2Sm{}, make_shape(Int<TileN>{}, Int<TileK>{})));

	using SmemLayoutAtom = UMMA::Layout_MN_SW128_Atom<Element>;
	using SmemLayoutDYT = decltype(UMMA::tile_to_mma_shape(
		SmemLayoutAtom{}, append(MmaShapeA_MK{}, Int<Stages>{}),
		Step<_2, _1, _3>{}));
	using SmemLayoutZ = decltype(UMMA::tile_to_mma_shape(
		SmemLayoutAtom{}, append(MmaShapeB_NK{}, Int<Stages>{}),
		Step<_2, _1, _3>{}));
	using SmemLayoutDYT_1 =
		decltype(SmemLayoutDYT{}(_, _, _, Int<0>{}));
	using SmemLayoutZ_1 =
		decltype(SmemLayoutZ{}(_, _, _, Int<0>{}));

	static constexpr int TmaTransBytesDYT =
		static_cast<int>(cosize_v<SmemLayoutDYT_1> * sizeof(Element));
	static constexpr int TmaTransBytesZ =
		static_cast<int>(cosize_v<SmemLayoutZ_1> * sizeof(Element));
	static constexpr int TmaTransBytes =
		2 * (TmaTransBytesDYT + TmaTransBytesZ);

	using MainloopPipelineUmma2Sm = cutlass::PipelineTmaUmmaAsync<
		Stages, ClusterShape, AtomThrShape>;
	using MainloopPipeline = MainloopPipelineUmma2Sm;
	using MainloopPipelineUmma = MainloopPipelineUmma2Sm;
	using PipelineState = typename MainloopPipelineUmma2Sm::PipelineState;
	using AccumulatorPipeline2Sm = cutlass::PipelineUmmaAsync<
		AccStages, AtomThrShape>;

	static constexpr int WarpSize = 32;
	static constexpr int WarpGroupSize = 128;
	static constexpr int NumConsumers = 2;
	static constexpr int ConsumerThreads = NumConsumers * WarpGroupSize;
	static constexpr int NumThreads = 384;

	using SmemLayoutStoreSlot = cute::conditional_t<
		CompactEpilogue,
		Layout<
			Shape<Int<AtomTileM>, Int<EpiChunkN>>,
			Stride<Int<EpiChunkN>, _1>>,
		Layout<
			Shape<Int<CtaTileM>, Int<EpiChunkN>>,
			Stride<Int<EpiChunkN>, _1>>>;
	using SmemLayoutStore = Layout<
		Shape<Int<AtomTileM>, Int<EpiChunkN>>,
		Stride<Int<EpiChunkN>, _1>>;
};

template <typename Traits>
struct Mlp3Smem2Sm {
	using Element = typename Traits::Element;

	static constexpr int smem_DYT_size =
		cosize_v<typename Traits::SmemLayoutDYT>;
	static constexpr int smem_Z_size =
		cosize_v<typename Traits::SmemLayoutZ>;
	static constexpr int smem_store_size =
		cosize_v<typename Traits::SmemLayoutStoreSlot>;

	alignas(128) Element smem_DYT[smem_DYT_size];
	alignas(128) Element smem_Z[smem_Z_size];
	alignas(128) Element store_buf[2 * smem_store_size];
	alignas(16) typename Traits::MainloopPipelineUmma2Sm::SharedStorage pipe_storage;
	alignas(16) uint32_t tmem_base;
	alignas(16) typename Traits::AccumulatorPipeline2Sm::SharedStorage acc_pipe;

	CUTE_DEVICE Element* DYT_data() { return &smem_DYT[0]; }
	CUTE_DEVICE Element* Z_data() { return &smem_Z[0]; }
};

template <typename Traits>
struct Mlp3FusedSmem2Sm {
	using Element = typename Traits::Element;

	static constexpr int smem_DYT_size =
		cosize_v<typename Traits::SmemLayoutDYT>;
	static constexpr int smem_Z_size =
		cosize_v<typename Traits::SmemLayoutZ>;
	static constexpr int smem_store_size =
		cosize_v<typename Traits::SmemLayoutStoreSlot>;

	alignas(128) Element smem_DYT[smem_DYT_size];
	alignas(128) Element smem_Z[smem_Z_size];
	alignas(128) Element store_buf[2 * smem_store_size];
	alignas(16) uint32_t tmem_base;
	alignas(16) typename Traits::AccumulatorPipeline2Sm::SharedStorage acc_pipe;

	CUTE_DEVICE Element* DYT_data() { return &smem_DYT[0]; }
	CUTE_DEVICE Element* Z_data() { return &smem_Z[0]; }
};

template <typename Traits>
__device__ __forceinline__ typename Traits::MainloopPipelineUmma2Sm
mlp3_make_pipe_umma_2sm(
		typename Traits::MainloopPipelineUmma2Sm::SharedStorage& storage) {
	using Pipeline = typename Traits::MainloopPipelineUmma2Sm;
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
			threadIdx.x == 0 &&
			cute::block_rank_in_cluster() % 2 == 0;
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
	typename Pipeline,
	typename SmemType,
	typename TmaLoadDYT,
	typename TmaLoadZ>
__device__ __forceinline__ void mlp3_producer_2sm(
		Pipeline& pipe,
		typename Traits::PipelineState& state,
		SmemType& smem,
		TmaLoadDYT const& tma_load_dyt,
		TmaLoadZ const& tma_load_z,
		const int* expert_k_starts,
		const int* expert_k_ends,
		int num_experts,
		int hidden_dim,
		int intermediate_dim,
		int num_tokens,
		int num_m_tiles,
		int num_n_tiles,
		int outer_split,
		int cell_start,
		int cell_stride,
		int batch_kb_start,
		int batch_kb_end,
		int k_split,
		int ring_kb = 0) {
	auto sDYT = make_tensor(
		make_smem_ptr(smem.DYT_data()), typename Traits::SmemLayoutDYT{});
	auto sZ = make_tensor(
		make_smem_ptr(smem.Z_data()), typename Traits::SmemLayoutZ{});

	auto mDYT = tma_load_dyt.get_tma_tensor(make_shape(
		static_cast<int64_t>(hidden_dim),
		static_cast<int64_t>(num_tokens)));
	auto mZT = tma_load_z.get_tma_tensor(make_shape(
		static_cast<int64_t>(intermediate_dim),
		static_cast<int64_t>(num_tokens)));

	typename Traits::TiledMma2Sm tiled_mma;
	int cta_rank = static_cast<int>(cute::block_rank_in_cluster());
	int pair_id = cta_rank / 2;
	int pair_rank = cta_rank % 2;
	auto cta_mma = tiled_mma.get_slice(pair_rank);
	using PairClusterShape = Shape<_2, _1, _1>;
	auto pair_layout_vmnk = tiled_divide(
		make_layout(PairClusterShape{}),
		make_tile(typename Traits::TiledMma2Sm::AtomThrID{}));
	auto pair_coord_vmnk =
		pair_layout_vmnk.get_flat_coord(pair_rank);
	auto cluster_layout_vmnk = tiled_divide(
		make_layout(typename Traits::ClusterShape{}),
		make_tile(typename Traits::TiledMma2Sm::AtomThrID{}));
	auto cluster_coord_vmnk =
		cluster_layout_vmnk.get_flat_coord(cta_rank);
	auto z_layout_vmnk = [&]() {
		if constexpr (Traits::MulticastShared)
			return cluster_layout_vmnk;
		else
			return pair_layout_vmnk;
	}();
	auto z_coord_vmnk = [&]() {
		if constexpr (Traits::MulticastShared)
			return cluster_coord_vmnk;
		else
			return pair_coord_vmnk;
	}();
	uint16_t mcast_mask_dyt =
		create_tma_multicast_mask<2>(pair_layout_vmnk, pair_coord_vmnk);
	uint16_t mcast_mask_z =
		create_tma_multicast_mask<1>(pair_layout_vmnk, pair_coord_vmnk);
	uint16_t cluster_mcast_mask_z =
		create_tma_multicast_mask<1>(
			cluster_layout_vmnk, cluster_coord_vmnk);

	int total_chunks = num_experts * num_n_tiles;
	int total_cells = total_chunks * outer_split * k_split;

	for (int cell_idx = cell_start;
	     cell_idx < total_cells;
	     cell_idx += cell_stride) {
		int k_slice = cell_idx % k_split;
		int cell_om = cell_idx / k_split;
		int chunk_idx = cell_om / outer_split;
		int lane = cell_om - chunk_idx * outer_split;

		int e = chunk_idx / num_n_tiles;
		int n_tile = chunk_idx - e * num_n_tiles;
		int pair_lane = lane * Traits::NumPairs + pair_id;
		int pair_splits = outer_split * Traits::NumPairs;
		int walk_begin = pair_lane * num_m_tiles / pair_splits;
		int walk_end = (pair_lane + 1) * num_m_tiles / pair_splits;

		int kb_lo = max(expert_k_starts[e], batch_kb_start);
		int kb_hi = min(expert_k_ends[e], batch_kb_end);
		if (kb_hi <= kb_lo)
			continue;

		int k_total = kb_hi - kb_lo;
		int k_per = (k_total + k_split - 1) / k_split;
		int slice_lo = kb_lo + k_slice * k_per;
		int slice_hi = min(slice_lo + k_per, kb_hi);
		if (slice_hi <= slice_lo)
			continue;
		kb_lo = slice_lo;
		kb_hi = slice_hi;

		for (int m_tile = walk_begin; m_tile < walk_end; ++m_tile) {
			auto coord = make_coord(m_tile, n_tile, _);
			auto gDYT = local_tile(
				mDYT, typename Traits::TileShape{}, coord,
				Step<_1, X, _1>{});
			auto gZ = local_tile(
				mZT, typename Traits::TileShape{}, coord,
				Step<X, _1, _1>{});
			auto tCgDYT = cta_mma.partition_A(gDYT);
			auto tCgZ = cta_mma.partition_B(gZ);

			auto [tDYTgDYT, tDYTsDYT] = tma_partition(
				tma_load_dyt,
				get<2>(pair_coord_vmnk),
				make_layout(size<2>(pair_layout_vmnk)),
				group_modes<0, 3>(sDYT),
				group_modes<0, 3>(tCgDYT));
			auto [tZgZ, tZsZ] = tma_partition(
				tma_load_z,
				get<1>(z_coord_vmnk),
				make_layout(size<1>(z_layout_vmnk)),
				group_modes<0, 3>(sZ),
				group_modes<0, 3>(tCgZ));

			for (int kb = kb_lo; kb < kb_hi; ++kb) {
				int kb_g = ring_kb > 0 ? kb % ring_kb : kb;
				pipe.producer_acquire(state);
				if (cute::elect_one_sync()) {
					auto* barrier = pipe.producer_get_barrier(state);
					if constexpr (Traits::ClusterM == 4) {
						copy(
							tma_load_dyt.with(*barrier),
							tDYTgDYT(_, kb_g),
							tDYTsDYT(_, state.index()));
						if constexpr (Traits::MulticastShared) {
							copy(
								tma_load_z.with(
									*barrier,
									cluster_mcast_mask_z),
								tZgZ(_, kb_g),
								tZsZ(_, state.index()));
						} else {
							copy(
								tma_load_z.with(*barrier),
								tZgZ(_, kb_g),
								tZsZ(_, state.index()));
						}
					} else {
						copy(
							tma_load_dyt.with(
								*barrier, mcast_mask_dyt),
							tDYTgDYT(_, kb_g),
							tDYTsDYT(_, state.index()));
						copy(
							tma_load_z.with(
								*barrier, mcast_mask_z),
							tZgZ(_, kb_g),
							tZsZ(_, state.index()));
					}
				}
				++state;
			}
		}
	}
	pipe.producer_tail(state);
}

template <
	typename Traits,
	typename Pipeline,
	typename SmemType,
	typename TmaReduceAddDA>
__device__ __forceinline__ void mlp3_consumer_2sm(
		Pipeline& pipe,
		typename Traits::PipelineState& state,
		SmemType& smem,
		TmaReduceAddDA const& tma_reduce_da,
		const int* expert_k_starts,
		const int* expert_k_ends,
		int num_experts,
		int intermediate_dim,
		int total_n_rows,
		int num_m_tiles,
		int num_n_tiles,
		int outer_split,
		int cell_start,
		int cell_stride,
		int batch_kb_start,
		int batch_kb_end,
		int k_split
#if defined(MLP3_2SM_COUNT_ACC_WAITS) && MLP3_2SM_COUNT_ACC_WAITS
		// Benchmark-only instrumentation (bench_opt/mlp3_s5_early_release/).
		// Only exists in translation units that define
		// MLP3_2SM_COUNT_ACC_WAITS=1 before including this header; every
		// production/timing build (moe_bwd.cu, mlp_bwd.cuh, bench_mlp3_2sm.cu)
		// leaves the macro undefined, so this parameter pair — and every use
		// of it below — is not even parsed there. Counts, from the leader
		// CTA's single MMA warp, how many accumulator producer_acquire calls
		// were already-satisfied (token == WaitDone) vs. required an actual
		// empty-barrier wait (token == WaitAgain), via producer_try_acquire.
		, unsigned long long* acc_acquire_total_counter = nullptr
		, unsigned long long* acc_acquire_wait_counter = nullptr
#endif
		) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
	using Element = typename Traits::Element;
	constexpr int TileM = Traits::TileM;
	constexpr int TileN = Traits::TileN;
	constexpr int CtaTileM = Traits::CtaTileM;
	constexpr int EpiChunkN = Traits::EpiChunkN;
	constexpr int NumEpiRounds = Traits::NumEpiRounds;
	constexpr int kMmaEpiThreads =
		Traits::ConsumerThreads + Traits::WarpSize;

	int cta_rank = static_cast<int>(cute::block_rank_in_cluster());
	int pair_id = cta_rank / 2;
	int pair_rank = cta_rank % 2;
	bool is_leader_cta = pair_rank == 0;
	int warp_id = threadIdx.x / Traits::WarpSize;
	bool is_mma_warp = warp_id == 3;
	bool is_epilogue = warp_id >= 4 && warp_id <= 11;
	int tid_in_epi = threadIdx.x - Traits::WarpGroupSize;
	int wg = is_epilogue ? tid_in_epi / Traits::WarpGroupSize : 0;
	int tid_in_wg = is_epilogue ? tid_in_epi % Traits::WarpGroupSize : 0;
	int wg_barrier_id = 1 + wg;
	bool is_wg_leader = is_epilogue && tid_in_wg == 0;

	typename Traits::TiledMma2Sm tiled_mma;
	auto cta_mma = tiled_mma.get_slice(pair_rank);
	auto sDYT = make_tensor(
		make_smem_ptr(smem.DYT_data()), typename Traits::SmemLayoutDYT{});
	auto sZ = make_tensor(
		make_smem_ptr(smem.Z_data()), typename Traits::SmemLayoutZ{});
	auto tCrDYT = cta_mma.make_fragment_A(sDYT);
	auto tCrZ = cta_mma.make_fragment_B(sZ);

	auto cAccFull = make_identity_tensor(
		make_shape(Int<TileM>{}, Int<TileN>{}));
	auto tCgC = cta_mma.partition_C(cAccFull);
	auto tCtAcc = cta_mma.make_fragment_C(tCgC);

	using AccPipe = typename Traits::AccumulatorPipeline2Sm;
	typename AccPipe::Params acc_params;
	acc_params.role = is_mma_warp && is_leader_cta
		? AccPipe::ThreadCategory::Producer
		: AccPipe::ThreadCategory::Consumer;
	acc_params.producer_arv_count = 1;
	acc_params.consumer_arv_count = 2;
	acc_params.initializing_warp = 4;
	AccPipe acc_pipe(
		smem.acc_pipe, acc_params, typename Traits::ClusterShape{});
	auto acc_prod_state =
		cutlass::make_producer_start_state<AccPipe>();
	typename AccPipe::PipelineState acc_cons_state;

	cutlass::arch::NamedBarrier::sync(kMmaEpiThreads, 3);
	uint32_t tmem_base = smem.tmem_base;
	tCtAcc.data() = tmem_base;

	constexpr int store_slot_rows =
		Traits::CompactEpilogue ? Traits::AtomTileM : CtaTileM;
	constexpr int store_slot_elems = store_slot_rows * EpiChunkN;
	Element* store_ptr = smem.store_buf + wg * store_slot_elems;
	auto sStore = make_tensor(
		make_smem_ptr(store_ptr),
		typename Traits::SmemLayoutStoreSlot{});

	auto mdA = tma_reduce_da.get_tma_tensor(make_shape(
		static_cast<int64_t>(total_n_rows),
		static_cast<int64_t>(intermediate_dim)));
	auto cta_tma_da = tma_reduce_da.get_slice(Int<0>{});

	auto epi_tile =
		make_tile(Int<CtaTileM>{}, Int<EpiChunkN>{});
	auto acc_mn = tCtAcc(make_coord(_, _), _0{}, _0{});
	auto tAcc_epi = flat_divide(acc_mn, epi_tile);
	auto t2r = make_tmem_copy(
		TmemLoadOp<EpiChunkN>{},
		tAcc_epi(_, _, _0{}, _0{}));
	auto thr_t2r = t2r.get_slice(tid_in_wg);
	auto cChunk = make_identity_tensor(
		make_shape(Int<CtaTileM>{}, Int<EpiChunkN>{}));
	auto tTR_cChunk = thr_t2r.partition_D(cChunk);
	auto tTR_rAcc = make_tensor<float>(shape(tTR_cChunk));

	Layout tmem_warp_layout =
		typename decltype(make_tmem_warp_partitioner(
			tAcc_epi(_, _, _0{}, _0{})))::TiledLayout_TV{};
	constexpr bool predicate_tmem_load =
		size(tmem_warp_layout) != cosize(tmem_warp_layout);

	int total_chunks = num_experts * num_n_tiles;
	int total_cells = total_chunks * outer_split * k_split;
	bool store_in_flight = false;

	for (int cell_idx = cell_start;
	     cell_idx < total_cells;
	     cell_idx += cell_stride) {
		int k_slice = cell_idx % k_split;
		int cell_om = cell_idx / k_split;
		int chunk_idx = cell_om / outer_split;
		int lane = cell_om - chunk_idx * outer_split;

		int e = chunk_idx / num_n_tiles;
		int n_tile = chunk_idx - e * num_n_tiles;
		int pair_lane = lane * Traits::NumPairs + pair_id;
		int pair_splits = outer_split * Traits::NumPairs;
		int walk_begin = pair_lane * num_m_tiles / pair_splits;
		int walk_end = (pair_lane + 1) * num_m_tiles / pair_splits;

		int kb_lo = max(expert_k_starts[e], batch_kb_start);
		int kb_hi = min(expert_k_ends[e], batch_kb_end);
		if (kb_hi <= kb_lo)
			continue;

		int k_total = kb_hi - kb_lo;
		int k_per = (k_total + k_split - 1) / k_split;
		int slice_lo = kb_lo + k_slice * k_per;
		int slice_hi = min(slice_lo + k_per, kb_hi);
		if (slice_hi <= slice_lo)
			continue;
		kb_lo = slice_lo;
		kb_hi = slice_hi;

		for (int m_tile = walk_begin; m_tile < walk_end; ++m_tile) {
			if (is_mma_warp && is_leader_cta) {
#if defined(MLP3_2SM_COUNT_ACC_WAITS) && MLP3_2SM_COUNT_ACC_WAITS
				// Non-blocking probe first (mbarrier.try_wait.parity):
				// tells us whether the stage was already free without
				// altering what producer_acquire() below actually does —
				// passing the resulting token into producer_acquire makes
				// it skip the redundant blocking .wait() when the token
				// already reports WaitDone, so behavior matches the
				// uninstrumented `producer_acquire(acc_prod_state)` call
				// exactly; only the counting is extra.
				auto acc_token = acc_pipe.producer_try_acquire(acc_prod_state);
				if (acc_acquire_total_counter != nullptr &&
				    (threadIdx.x % Traits::WarpSize) == 0)
					atomicAdd(acc_acquire_total_counter, 1ull);
				if (acc_token.get() == cutlass::BarrierStatus::WaitAgain &&
				    acc_acquire_wait_counter != nullptr &&
				    (threadIdx.x % Traits::WarpSize) == 0)
					atomicAdd(acc_acquire_wait_counter, 1ull);
				acc_pipe.producer_acquire(acc_prod_state, acc_token);
#else
				acc_pipe.producer_acquire(acc_prod_state);
#endif
				int acc_stage = acc_prod_state.index();
				tCtAcc.data() =
					tmem_base + uint32_t(acc_stage * TileN);

				bool first = true;
				for (int kb = kb_lo; kb < kb_hi; ++kb) {
					pipe.consumer_wait(state);
					CUTE_UNROLL
					for (int ks = 0; ks < size<2>(tCrDYT); ++ks) {
						tiled_mma.accumulate_ = first
							? UMMA::ScaleOut::Zero
							: UMMA::ScaleOut::One;
						first = false;
						gemm(
							tiled_mma,
							tCrDYT(_, _, ks, state.index()),
							tCrZ(_, _, ks, state.index()),
							tCtAcc);
					}
					pipe.consumer_release(state);
					++state;
				}
				acc_pipe.producer_commit(acc_prod_state);
				++acc_prod_state;
			}

			if (is_epilogue) {
				acc_pipe.consumer_wait(acc_cons_state);
				int acc_stage = acc_cons_state.index();
				tCtAcc.data() =
					tmem_base + uint32_t(acc_stage * TileN);
				auto acc_mn_stage =
					tCtAcc(make_coord(_, _), _0{}, _0{});
				auto tAcc_epi_stage =
					flat_divide(acc_mn_stage, epi_tile);
				auto tTR_tAcc =
					thr_t2r.partition_S(tAcc_epi_stage);

				CUTE_UNROLL
				for (int round = 0;
				     round < NumEpiRounds;
				     ++round) {
					int chunk = wg * NumEpiRounds + round;
					auto tAccChunk =
						tTR_tAcc(_, _, _, _0{}, chunk);
					bool issue_tmem_load = true;
					if constexpr (predicate_tmem_load) {
						int subpart =
							(tAccChunk.data().dp_ / 32) % 4;
						issue_tmem_load =
							tid_in_wg / Traits::WarpSize == subpart;
					}
					if (issue_tmem_load)
						copy(t2r, tAccChunk, tTR_rAcc);

					if constexpr (Traits::EarlyTmemRelease) {
						// Early-TMEM-release experiment (default off;
						// gated by Traits::EarlyTmemRelease — see the
						// EarlyTmemRelease_ trait param doc comment).
						//
						// Safety argument (see 2sm_stage.md / the
						// mlp3_s5_early_release campaign notes for the
						// full audit):
						//  1. tAccChunk/TMEM is never read again after
						//     this point: rounds execute in strict
						//     per-thread program order (CUTE_UNROLL keeps
						//     the source order; it only controls
						//     unrolling), so by the time a thread reaches
						//     the *last* round's copy(), every earlier
						//     round's copy() already retired for that
						//     same thread. All remaining work in this
						//     round (below) reads only tTR_rAcc
						//     (registers) and sStore (SMEM) — never
						//     tCtAcc/TMEM again.
						//  2. predicate_tmem_load only skips the copy()
						//     call itself for warps outside the active
						//     subpartition; every warp (issuing or not,
						//     this round or an earlier one) still reaches
						//     this same program point and participates in
						//     the barrier below, so the rendezvous still
						//     covers every TMEM-reading warp in both
						//     epilogue warpgroups.
						//  3. The 256-thread NamedBarrier (id 0, unchanged
						//     from the default post-loop placement) is
						//     CTA-local and gates only "this CTA's own
						//     epilogue threads are done reading". Cross-
						//     CTA safety does not depend on it: consumer_
						//     release() on AccPipe = PipelineUmmaAsync<
						//     AccStages, AtomThrShape=(2,1,1)> lowers (for
						//     a 2-CTA atom) to consumer_release_2x1SM ->
						//     umma_arrive_2x1SM_sm0, a native cluster-
						//     scope mbarrier.arrive that always targets
						//     the leader ("sm0") CTA's empty barrier
						//     regardless of which CTA issues it. The
						//     pipeline is constructed with
						//     consumer_arv_count=2 (one arrival expected
						//     per CTA), so the producer's next
						//     producer_acquire only unblocks once BOTH
						//     CTAs' elected threads have arrived — each
						//     gated behind its own CTA-local rendezvous.
						//     Moving the call earlier changes only when
						//     each CTA arrives, not the aggregation
						//     contract, so paired-CTA safety is preserved
						//     unmodified.
						//  4. Every epilogue thread (not just tid_in_epi
						//     ==0) still advances acc_cons_state exactly
						//     once per m_tile — same cadence as the
						//     default post-loop release it replaces.
						if (round == NumEpiRounds - 1) {
							cutlass::arch::NamedBarrier::sync(
								Traits::ConsumerThreads, 0);
							if (tid_in_epi == 0)
								acc_pipe.consumer_release(
									acc_cons_state);
							++acc_cons_state;
						}
					}

					int da_m = e * num_m_tiles + m_tile;
					int n_chunk =
						n_tile *
							(Traits::TileN / Traits::EpiChunkN) +
						wg * Traits::NumEpiRounds + round;

					if constexpr (Traits::CompactEpilogue) {
						// Reuse one 64xEpiChunkN TMA box per warpgroup for
						// the CTA's two 64-row atoms. The TMEM tile is loaded
						// once; only the register-to-SMEM fold and the two
						// reduce-add stores are serialized.
						CUTE_UNROLL
						for (int atom = 0;
						     atom < Traits::kAtomsPerCta;
						     ++atom) {
							if (store_in_flight)
								cute::tma_store_wait<0>();
							cutlass::arch::NamedBarrier::sync(
								Traits::WarpGroupSize,
								wg_barrier_id);

							if (issue_tmem_load) {
								CUTE_UNROLL
								for (int i = 0;
								     i < size(tTR_rAcc);
								     ++i) {
									int m_local =
										get<0>(tTR_cChunk(i));
									int n_local =
										get<1>(tTR_cChunk(i));
									if (m_local /
											Traits::AtomTileM ==
									    atom) {
										sStore(
											m_local -
												atom *
													Traits::AtomTileM,
											n_local) =
											static_cast<Element>(
												tTR_rAcc(i));
									}
								}
							}
							cutlass::arch::NamedBarrier::sync(
								Traits::WarpGroupSize,
								wg_barrier_id);

							if (is_wg_leader) {
								cute::tma_store_fence();
								int m_atom =
									(Traits::TileM /
									 Traits::AtomTileM) *
										da_m +
									Traits::kAtomsPerCta *
										pair_rank +
									atom;
								auto gdA = local_tile(
									mdA,
									make_tile(
										Int<Traits::AtomTileM>{},
										Int<Traits::EpiChunkN>{}),
									make_coord(m_atom, n_chunk));
								copy(
									tma_reduce_da,
									cta_tma_da.partition_S(sStore),
									cta_tma_da.partition_D(gdA));
								cute::tma_store_arrive();
							}
							store_in_flight = true;
						}
					} else {
						if (store_in_flight)
							cute::tma_store_wait<0>();
						cutlass::arch::NamedBarrier::sync(
							Traits::WarpGroupSize,
							wg_barrier_id);

						if (issue_tmem_load) {
							CUTE_UNROLL
							for (int i = 0;
							     i < size(tTR_rAcc);
							     ++i) {
								int m_local =
									get<0>(tTR_cChunk(i));
								int n_local =
									get<1>(tTR_cChunk(i));
								sStore(m_local, n_local) =
									static_cast<Element>(
										tTR_rAcc(i));
							}
						}
						cutlass::arch::NamedBarrier::sync(
							Traits::WarpGroupSize,
							wg_barrier_id);

						if (is_wg_leader) {
							cute::tma_store_fence();
							CUTE_UNROLL
							for (int atom = 0;
							     atom < Traits::kAtomsPerCta;
							     ++atom) {
								int m_atom =
									(Traits::TileM /
									 Traits::AtomTileM) *
										da_m +
									Traits::kAtomsPerCta *
										pair_rank +
									atom;
								auto sStoreAtom =
									local_tile(
										sStore,
										make_tile(
											Int<Traits::AtomTileM>{},
											Int<Traits::EpiChunkN>{}),
										make_coord(atom, 0));
								auto gdA = local_tile(
									mdA,
									make_tile(
										Int<Traits::AtomTileM>{},
										Int<Traits::EpiChunkN>{}),
									make_coord(m_atom, n_chunk));
								copy(
									tma_reduce_da,
									cta_tma_da.partition_S(
										sStoreAtom),
									cta_tma_da.partition_D(gdA));
							}
							cute::tma_store_arrive();
						}
						store_in_flight = true;
					}
				}

				if constexpr (!Traits::EarlyTmemRelease) {
					// Current/default behavior (A): release only after
					// every round's register->SMEM fold and
					// TMA_REDUCE_ADD store work has been issued for this
					// m_tile. Skipped entirely when EarlyTmemRelease is
					// set, since that path already released (and
					// advanced acc_cons_state) inside the loop above,
					// right after the final round's synchronous TMEM
					// read — see the EarlyTmemRelease branch for the
					// safety argument. Exactly one of the two paths runs
					// per m_tile, so acc_cons_state still advances
					// exactly once either way.
					cutlass::arch::NamedBarrier::sync(
						Traits::ConsumerThreads, 0);
					if (tid_in_epi == 0)
						acc_pipe.consumer_release(acc_cons_state);
					++acc_cons_state;
				}
			}
		}
	}

	if (is_epilogue && store_in_flight)
		cute::tma_store_wait<0>();
	cutlass::arch::NamedBarrier::sync(kMmaEpiThreads, 3);
#else
	__trap();
#endif
}

template <
	typename Traits,
	typename TmaLoadDYT,
	typename TmaLoadZ,
	typename TmaReduceAddDA>
__device__ __forceinline__ void mlp3_fwd_2sm(
		Mlp3Smem2Sm<Traits>& smem,
		TmaLoadDYT const& tma_load_dyt,
		TmaLoadZ const& tma_load_z,
		TmaReduceAddDA const& tma_reduce_da,
		const int* expert_k_starts,
		const int* expert_k_ends,
		int num_experts,
		int hidden_dim,
		int intermediate_dim,
		int num_tokens,
		int num_m_tiles,
		int num_n_tiles,
		int outer_split,
		int batch_kb_start = 0,
		int batch_kb_end = -1,
		int k_split = 1,
		int ring_kb = 0) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
	if (gridDim.x % Traits::ClusterM != 0)
		__trap();
	if constexpr (Traits::NumPairs > 1) {
		if (num_m_tiles % (outer_split * Traits::NumPairs) != 0)
			__trap();
	}

	using Pipeline = typename Traits::MainloopPipelineUmma2Sm;
	using PipeState = typename Traits::PipelineState;

	int warp_id = threadIdx.x / Traits::WarpSize;
	bool is_producer = warp_id == 0;
	bool is_consumer = warp_id >= 3 && warp_id <= 11;

	cute::prefetch_tma_descriptor(
		tma_load_dyt.get_tma_descriptor());
	cute::prefetch_tma_descriptor(
		tma_load_z.get_tma_descriptor());
	cute::prefetch_tma_descriptor(
		tma_reduce_da.get_tma_descriptor());

	auto pipe =
		mlp3_make_pipe_umma_2sm<Traits>(smem.pipe_storage);
	cute::cluster_sync();

	cute::TMEM::Allocator2Sm tmem_allocator;
	constexpr int kTmemColumns =
		Traits::AccStages * Traits::TileN;
	if (warp_id == 4) {
		tmem_allocator.allocate(kTmemColumns, &smem.tmem_base);
		__syncwarp();
	}
	__syncthreads();
	cute::cluster_sync();

	PipeState producer_state = is_producer
		? cutlass::make_producer_start_state<Pipeline>()
		: PipeState{};
	PipeState consumer_state;

	int cell_start =
		static_cast<int>(blockIdx.x) / Traits::ClusterM;
	int cell_stride =
		static_cast<int>(gridDim.x) / Traits::ClusterM;
	int kb_end = batch_kb_end >= 0
		? batch_kb_end
		: num_tokens / Traits::TileK;

	if (is_producer) {
		mlp3_producer_2sm<Traits>(
			pipe,
			producer_state,
			smem,
			tma_load_dyt,
			tma_load_z,
			expert_k_starts,
			expert_k_ends,
			num_experts,
			hidden_dim,
			intermediate_dim,
			num_tokens,
			num_m_tiles,
			num_n_tiles,
			outer_split,
			cell_start,
			cell_stride,
			batch_kb_start,
			kb_end,
			k_split,
			ring_kb);
	} else if (is_consumer) {
		mlp3_consumer_2sm<Traits>(
			pipe,
			consumer_state,
			smem,
			tma_reduce_da,
			expert_k_starts,
			expert_k_ends,
			num_experts,
			intermediate_dim,
			num_experts * num_m_tiles * Traits::TileM,
			num_m_tiles,
			num_n_tiles,
			outer_split,
			cell_start,
			cell_stride,
			batch_kb_start,
			kb_end,
			k_split);
	}

	__syncthreads();
	cute::cluster_sync();
	if (warp_id == 4) {
		tmem_allocator.release_allocation_lock();
		tmem_allocator.free(smem.tmem_base, kTmemColumns);
	}
#else
	__trap();
#endif
}

} // namespace liger
