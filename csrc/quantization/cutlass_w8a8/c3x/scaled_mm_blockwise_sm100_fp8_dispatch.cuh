#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/group_array_problem_shape.hpp"


#include "cutlass_extensions/gemm/dispatch_policy.hpp"
#include "cutlass_extensions/gemm/collective/collective_builder.hpp"

#include "cutlass_gemm_caller.cuh"

namespace vllm {

using namespace cutlass::gemm::collective;

template <typename OutType, int GroupSizeM_, int GroupSizeN_, int GroupSizeK_,
          int TileSizeM_ = 128, class ClusterShape = Shape<_1, _2, _1>>
struct cutlass_3x_gemm_fp8_blockwise {
  using GroupSizeM = Int<GroupSizeM_>;
  using GroupSizeN = Int<GroupSizeN_>;
  using GroupSizeK = Int<GroupSizeK_>;
  using TileSizeM = Int<TileSizeM_>;

  static_assert(TileSizeM_ % GroupSizeM_ == 0,
                "TileSizeM must be a multiple of GroupSizeM");

  using ElementAB = cutlass::float_e4m3_t;
  using ElementBlockScale = cutlass::float_ue8m0_t;

  using ElementPairA = cute::tuple<ElementAB, ElementBlockScale>;
  using LayoutA = cutlass::layout::RowMajor;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementAB>::value;

  using ElementPairB = cute::tuple<ElementAB, ElementBlockScale>;
  using LayoutB = cutlass::layout::ColumnMajor;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementAB>::value;

  using ElementD = OutType;
  using StrideD = Stride<int64_t, Int<1>, Int<0>>;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

  using ElementC = void;
  using StrideC = StrideD;
  static constexpr int AlignmentC = AlignmentD;

  using ElementAccumulator = float;
  using ElementCompute = float;
  using ArchTag = cutlass::arch::Sm100;
  using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using TileShape = Shape<TileSizeM, GroupSizeN, GroupSizeK>;

  // MainloopSm100TmaUmmaWarpSpecializedBlockScaled
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized1SmBlockScaledSm100;
  using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1Sm;
  using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;


//  using StoreEpilogueCompute = typename cutlass::epilogue::fusion::Sm90EVT<
//      cutlass::epilogue::fusion::Sm90AccFetch>;

  using CollectiveEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, OperatorClass, TileShape, ClusterShape, EpilogueTileType,
          ElementAccumulator, ElementCompute, ElementC, StrideC, AlignmentC,
          ElementD, StrideD, AlignmentD, EpilogueSchedule
          >::CollectiveOp;
  // removed , StoreEpilogueCompute

  using CollectiveMainloop =
      typename cutlass::gemm::collective::CollectiveBuilder<
          ArchTag, OperatorClass, ElementPairA, LayoutA, AlignmentA, ElementPairB,
          LayoutB, AlignmentB, ElementAccumulator, TileShape, ClusterShape,
          cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
              sizeof(typename CollectiveEpilogue::SharedStorage))>,
          KernelSchedule>::CollectiveOp;

  using KernelType = enable_sm90_or_later<cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>>;

  struct GemmKernel : public KernelType {};

  using StrideA = typename GemmKernel::StrideA;
  using StrideB = typename GemmKernel::StrideB;
};

template <typename Gemm>
void cutlass_gemm_caller_blockwise(torch::Tensor& out, torch::Tensor const& a,
                                   torch::Tensor const& b,
                                   torch::Tensor const& a_scales,
                                   torch::Tensor const& b_scales) {
  using GemmKernel = typename Gemm::GemmKernel;

  using ElementAB = typename Gemm::ElementAB;
  using ElementD = typename Gemm::ElementD;

  // Scale Factor tensors have an interleaved layout. Bring Layout instead of stride.
  using LayoutSFA = typename GemmKernel::CollectiveMainloop::LayoutSFA;
  using LayoutSFB = typename GemmKernel::CollectiveMainloop::LayoutSFB;
  using Sm100BlkScaledConfig =  typename Gemm::GemmKernel::CollectiveMainloop::Sm100BlkScaledConfig;

  auto prob_shape = c3x::get_problem_shape(a, b);
  int32_t m = get<0>(prob_shape), n = get<1>(prob_shape),
          k = get<2>(prob_shape);

  LayoutSFA layout_SFA = Sm100BlkScaledConfig::tile_atom_to_shape_SFA(prob_shape);
  LayoutSFB layout_SFB = Sm100BlkScaledConfig::tile_atom_to_shape_SFB(prob_shape);

  int64_t lda = a.stride(0);
  int64_t ldb = b.stride(1);
  int64_t ldc = out.stride(0);

  using StrideA = Stride<int64_t, Int<1>, int64_t>;
  using StrideB = Stride<int64_t, Int<1>, int64_t>;
  using StrideC = typename Gemm::StrideC;

  StrideA a_stride{lda, Int<1>{}, 0};
  StrideB b_stride{ldb, Int<1>{}, 0};
  StrideC c_stride{ldc, Int<1>{}, Int<0>{}};

  auto a_ptr = static_cast<ElementAB*>(a.data_ptr());
  auto b_ptr = static_cast<ElementAB*>(b.data_ptr());
  auto a_scales_ptr = static_cast<cutlass::float_ue8m0_t*>(a_scales.data_ptr());
  auto b_scales_ptr = static_cast<cutlass::float_ue8m0_t*>(b_scales.data_ptr());

  // Check is the t is contiguous and is 1D or 2D with one of the dimensions
  // being 1 (i.e. a row or column vector)
  auto is_contiguous_vector = [](const torch::Tensor& t) {
    auto t_sizes = t.sizes();
    return t.is_contiguous() &&
           (t.dim() == 1 ||
            (t.dim() == 2 &&
             *std::min_element(t_sizes.begin(), t_sizes.end()) == 1));
  };

  // TODO(lucas): lets clean-up the kernel so that we pass in Strides so
  //  we don't have to deal with enforcing implicit layouts
  TORCH_CHECK(a_scales.size(0) == m / Gemm::GroupSizeM::value);
  TORCH_CHECK(a_scales.size(1) == k / Gemm::GroupSizeK::value);
  TORCH_CHECK(a_scales.stride(0) == 1 || is_contiguous_vector(a_scales),
              "a_scales must be M major");
  TORCH_CHECK(b_scales.size(0) == k / Gemm::GroupSizeK::value);
  TORCH_CHECK(b_scales.size(1) == n / Gemm::GroupSizeN::value);
  TORCH_CHECK(b_scales.stride(0) == 1 || is_contiguous_vector(b_scales),
              "b_scales must be K major");
  typename GemmKernel::MainloopArguments mainloop_args{
      a_ptr, a_stride, b_ptr, b_stride, a_scales_ptr, layout_SFA, b_scales_ptr, layout_SFB};

  auto c_ptr = static_cast<ElementD*>(out.data_ptr());
  typename GemmKernel::EpilogueArguments epilogue_args{
      {}, c_ptr, c_stride, c_ptr, c_stride};

  c3x::cutlass_gemm_caller<GemmKernel>(a.device(), prob_shape, mainloop_args,
                                       epilogue_args);
}

template <typename OutType>
void cutlass_gemm_blockwise_sm100_fp8_dispatch(torch::Tensor& out,
                                              torch::Tensor const& a,
                                              torch::Tensor const& b,
                                              torch::Tensor const& a_scales,
                                              torch::Tensor const& b_scales) {
  cutlass_gemm_caller_blockwise<
      cutlass_3x_gemm_fp8_blockwise<OutType, 1, 128, 128>>(out, a, b, a_scales,
                                                           b_scales);
}

}  // namespace vllm