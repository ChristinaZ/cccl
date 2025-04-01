// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * @file
 *   cub::DeviceTopK provides device-wide, parallel operations for finding the K largest (or smallest) items
 * from sequences of unordered data items residing within device-accessible memory.
 */

#pragma once

#include <cub/config.cuh>

#include <cub/agent/agent_topk.cuh>
#include <cub/block/block_load.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_temporary_storage.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/cmath>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

CUB_NAMESPACE_BEGIN

namespace detail
{

namespace topk
{

#define DEFAULT_NUM_THREADS 512
template <class KeyT>
constexpr int calc_bits_per_pass()
{
  return sizeof(KeyT) == 1 ? 8 : sizeof(KeyT) == 2 ? 11 : sizeof(KeyT) == 4 ? 11 : sizeof(KeyT) == 8 ? 11 : 8;
}

template <class KeyInT>
struct sm90_tuning
{
  static constexpr int threads = DEFAULT_NUM_THREADS; // Number of threads per block

  static constexpr int nominal_4b_items_per_thread = 4;
  static constexpr int items = cuda::std::max(1, (nominal_4b_items_per_thread * 4 / (int) sizeof(KeyInT)));
  // Try to load 16 Bytes per thread. (int64(items=2);int32(items=4);int16(items=8)).

  static constexpr int BITS_PER_PASS          = detail::topk::calc_bits_per_pass<KeyInT>();
  static constexpr int COEFFICIENT_FOR_BUFFER = 128;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_VECTORIZE;
};

} // namespace topk
} // namespace detail

template <class KeyInT, class NumItemsT>
struct device_topk_policy_hub
{
  struct DefaultTuning
  {
    static constexpr int NOMINAL_4B_ITEMS_PER_THREAD = 4;
    static constexpr int ITEMS_PER_THREAD            = cuda::std::min(
      NOMINAL_4B_ITEMS_PER_THREAD, cuda::std::max(1, (NOMINAL_4B_ITEMS_PER_THREAD * 4 / (int) sizeof(KeyInT))));

    static constexpr int BITS_PER_PASS          = detail::topk::calc_bits_per_pass<KeyInT>();
    static constexpr int COEFFICIENT_FOR_BUFFER = 128;

    using TopKPolicyT =
      AgentTopKPolicy<DEFAULT_NUM_THREADS,
                      ITEMS_PER_THREAD,
                      BITS_PER_PASS,
                      COEFFICIENT_FOR_BUFFER,
                      BLOCK_LOAD_VECTORIZE,
                      BLOCK_SCAN_WARP_SCANS>;
  };

  struct Policy350
      : DefaultTuning
      , ChainedPolicy<350, Policy350, Policy350>
  {};

  struct Policy900 : ChainedPolicy<900, Policy900, Policy350>
  {
    using tuning = detail::topk::sm90_tuning<KeyInT>;

    using TopKPolicyT =
      AgentTopKPolicy<tuning::threads,
                      tuning::items,
                      tuning::BITS_PER_PASS,
                      tuning::COEFFICIENT_FOR_BUFFER,
                      tuning::load_algorithm,
                      BLOCK_SCAN_WARP_SCANS>;
  };

  using MaxPolicy = Policy900;
};
/******************************************************************************
 * Kernel entry points
 *****************************************************************************/
/**
 * TopK kernel entry point (multi-block) for histogram collection, prefix sum and filering operations except for the
 * last round
 *
 * Find the largest (or smallest) K items from a sequence of unordered data
 * @tparam KeyInputIteratorT
 *   **[inferred]** Random-access input iterator type for reading input keys @iterator
 *
 * @tparam KeyOutputIteratorT
 *   **[inferred]** Random-access output iterator type for writing output keys @iterator
 *
 * @tparam ValueInputIteratorT
 *   **[inferred]** Random-access input iterator type for reading input values @iterator
 *
 * @tparam ValueOutputIteratorT
 *   **[inferred]** Random-access output iterator type for writing output values @iterator
 *
 * @tparam NumItemsT
 *  Data Type for variables: num_items and k
 *
 * @tparam KeyInT
 *  Data Type for input keys
 *
 * @tparam ExtractBinOpT
 *   Operations to extract the bin from the input key value
 *
 * @tparam IdentifyCandidatesOpT
 *   Operations to filter the input key value
 *
 * @tparam SelectMin
 *   Determine whether to select the smallest (SelectMin=true) or largest (SelectMin=false) K elements.
 *
 * @param[in] d_keys_in
 *   Pointer to the input data of key data
 *
 * @param[out] d_keys_out
 *   Pointer to the K output sequence of key data
 *
 * @param[in] d_values_in
 *   Pointer to the input sequence of associated value items
 *
 * @param[out] d_values_out
 *   Pointer to the output sequence of associated value items
 *
 * @param[in] in_buf
 *   Pointer to buffer of input key data
 *
 * @param[out] out_buf
 *   Pointer to buffer of output key data
 *
 * @param[in] in_idx_buf
 *   Pointer to buffer of index of input buffer
 *
 * @param[out] out_idx_buf
 *   Pointer to buffer of index of output
 *
 * @param[in] counter
 *   Pointer to buffer of counter array
 *
 * @param[in] histogram
 *   Pointer to buffer of histogram array
 *
 * @param[in] num_items
 *   Number of items to be processed
 *
 * @param[in] k
 *   The K value. Will find K elements from num_items elements
 *
 * @param[in] extract_bin_op
 *   Extract the bin operator
 *
 * @param[in] identify_candidates_op
 *   Extract element filter operator
 *
 * @param[in] pass
 *   The index of the passes
 */
template <typename ChainedPolicyT,
          typename KeyInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueInputIteratorT,
          typename ValueOutputIteratorT,
          typename NumItemsT,
          typename KeyInT,
          typename ExtractBinOpT,
          typename IdentifyCandidatesOpT,
          bool SelectMin,
          bool IsFirstPass>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::TopKPolicyT::BLOCK_THREADS))
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceTopKKernel(
    const KeyInputIteratorT d_keys_in,
    KeyOutputIteratorT d_keys_out,
    const ValueInputIteratorT d_values_in,
    ValueOutputIteratorT d_values_out,
    KeyInT* in_buf,
    NumItemsT* in_idx_buf,
    KeyInT* out_buf,
    NumItemsT* out_idx_buf,
    Counter<detail::it_value_t<KeyInputIteratorT>, NumItemsT>* counter,
    NumItemsT* histogram,
    NumItemsT num_items,
    NumItemsT k,
    ExtractBinOpT extract_bin_op,
    IdentifyCandidatesOpT identify_candidates_op,
    int pass)
{
  using AgentTopKPolicyT = typename ChainedPolicyT::ActivePolicy::TopKPolicyT;
  using AgentTopKT =
    AgentTopK<AgentTopKPolicyT,
              KeyInputIteratorT,
              KeyOutputIteratorT,
              ValueInputIteratorT,
              ValueOutputIteratorT,
              ExtractBinOpT,
              IdentifyCandidatesOpT,
              NumItemsT,
              SelectMin>;

  // Shared memory storage
  __shared__ typename AgentTopKT::TempStorage temp_storage;
  AgentTopKT(
    temp_storage, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, k, extract_bin_op, identify_candidates_op)
    .InvokeOneSweep<IsFirstPass>(in_buf, in_idx_buf, out_buf, out_idx_buf, counter, histogram, pass);
}

/**
 * TopK kernel entry point for the last filtering step (multi-block)
 *
 * Find the largest (or smallest) K items from a sequence of unordered data
 * @tparam KeyInputIteratorT
 *   **[inferred]** Random-access input iterator type for reading input keys @iterator
 *
 * @tparam KeyOutputIteratorT
 *   **[inferred]** Random-access output iterator type for writing output keys @iterator
 *
 * @tparam ValueInputIteratorT
 *   **[inferred]** Random-access input iterator type for reading input values @iterator
 *
 * @tparam ValueOutputIteratorT
 *   **[inferred]** Random-access output iterator type for writing output values @iterator
 *
 * @tparam NumItemsT
 *  Data Type for variables: num_items and k
 *
 * @tparam KeyInT
 *  Data Type for input keys
 *
 * @tparam IdentifyCandidatesOpT
 *   Operations to filter the input key value
 *
 * @tparam SelectMin
 *   Determine whether to select the smallest (SelectMin=true) or largest (SelectMin=false) K elements.
 *
 * @param[in] d_keys_in
 *   Pointer to the input data of key data
 *
 * @param[out] d_keys_out
 *   Pointer to the K output sequence of key data
 *
 * @param[in] d_values_in
 *   Pointer to the input sequence of associated value items
 *
 * @param[out] d_values_out
 *   Pointer to the output sequence of associated value items
 *
 * @param[in] in_buf
 *   Pointer to buffer of input key data
 *
 * @param[in] in_idx_buf
 *   Pointer to buffer of index of input buffer
 *
 * @param[in] counter
 *   Pointer to buffer of counter array
 *
 * @param[in] histogram
 *   Pointer to buffer of histogram array
 *
 * @param[in] num_items
 *   Number of items to be processed
 *
 * @param[in] k
 *   The K value. Will find K elements from num_items elements
 *
 * @param[in] identify_candidates_op
 *   Extract element filter operator
 *
 * @param[in] pass
 *   The index of the passes
 */
template <typename ChainedPolicyT,
          typename KeyInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueInputIteratorT,
          typename ValueOutputIteratorT,
          typename NumItemsT,
          typename KeyInT,
          typename IdentifyCandidatesOpT,
          bool SelectMin>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::TopKPolicyT::BLOCK_THREADS))
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceTopKLastFilterKernel(
    const KeyInputIteratorT d_keys_in,
    KeyOutputIteratorT d_keys_out,
    const ValueInputIteratorT d_values_in,
    ValueOutputIteratorT d_values_out,
    KeyInT* in_buf,
    NumItemsT* in_idx_buf,
    Counter<detail::it_value_t<KeyInputIteratorT>, NumItemsT>* counter,
    NumItemsT* histogram,
    NumItemsT num_items,
    NumItemsT k,
    IdentifyCandidatesOpT identify_candidates_op,
    int pass)
{
  using AgentTopKPolicyT = typename ChainedPolicyT::ActivePolicy::TopKPolicyT;
  using ExtractBinOpT    = NullType;
  using AgentTopKT =
    AgentTopK<AgentTopKPolicyT,
              KeyInputIteratorT,
              KeyOutputIteratorT,
              ValueInputIteratorT,
              ValueOutputIteratorT,
              ExtractBinOpT, // ExtractBinOp operator (not used)
              IdentifyCandidatesOpT,
              NumItemsT,
              SelectMin>;

  // Shared memory storage
  __shared__ typename AgentTopKT::TempStorage temp_storage;
  AgentTopKT(
    temp_storage, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, k, ExtractBinOpT{}, identify_candidates_op)
    .InvokeLastFilter(in_buf, in_idx_buf, counter, histogram, k, pass);
}

/*
 * @tparam KeyInputIteratorT
 *   **[inferred]** Random-access input iterator type for reading input keys @iterator
 *
 * @tparam KeyOutputIteratorT
 *   **[inferred]** Random-access output iterator type for writing output keys @iterator
 *
 * @tparam ValueInputIteratorT
 *   **[inferred]** Random-access input iterator type for reading input values @iterator
 *
 * @tparam ValueOutputIteratorT
 *   **[inferred]** Random-access input iterator type for writing output values @iterator
 *
 * @tparam NumItemsT
 *  Type of variable num_items and k
 *
 * @tparam SelectMin
 *   Determine whether to select the smallest (SelectMin=true) or largest (SelectMin=false) K elements.
 *
 * @param[in] d_temp_storage
 *   Device-accessible allocation of temporary storage. When `nullptr`, the
 *   required allocation size is written to `temp_storage_bytes` and no work is done.
 *
 * @param[in,out] temp_storage_bytes
 *   Reference to size in bytes of `d_temp_storage` allocation
 *
 * @param[in] d_keys_in
 *   Pointer to the input data of key data to find top K
 *
 * @param[out] d_keys_out
 *   Pointer to the K output sequence of key data
 *
 * @param[in] d_values_in
 *   Pointer to the input sequence of associated value items
 *
 * @param[out] d_values_out
 *   Pointer to the output sequence of associated value items
 *
 * @param[in] num_items
 *   Number of items to be processed
 *
 * @param[in] k
 *   The K value. Will find K elements from num_items elements
 *
 * @param[in] stream
 *   @rst
 *   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
 *   @endrst
 */
template <typename KeyInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueInputIteratorT,
          typename ValueOutputIteratorT,
          typename NumItemsT,
          bool SelectMin,
          typename SelectedPolicy = device_topk_policy_hub<detail::it_value_t<KeyInputIteratorT>, NumItemsT>>
struct DispatchTopK : SelectedPolicy
{
  /// Device-accessible allocation of temporary storage.
  /// When `nullptr`, the required allocation size is written to `temp_storage_bytes`
  /// and no work is done.
  void* d_temp_storage;

  /// Reference to size in bytes of `d_temp_storage` allocation
  size_t& temp_storage_bytes;

  /// Pointer to the input sequence of data items
  KeyInputIteratorT d_keys_in;

  /// Pointer to the K output sequence of key data
  KeyOutputIteratorT d_keys_out;

  /// Pointer to the input sequence of associated value items
  ValueInputIteratorT d_values_in;

  /// Pointer to the output sequence of associated value items
  ValueOutputIteratorT d_values_out;

  /// Number of items to be processed
  NumItemsT num_items;

  /// The K value. Will find K elements from num_items elements
  NumItemsT k;

  /// CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
  cudaStream_t stream;

  int ptx_version;

  using KeyInT                    = detail::it_value_t<KeyInputIteratorT>;
  static constexpr bool KEYS_ONLY = std::is_same<ValueInputIteratorT, NullType>::value;
  /*
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to `temp_storage_bytes` and no work is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_keys_in
   *   Pointer to the input data of key data
   *
   * @param[out] d_keys_out
   *   Pointer to the K output sequence of key data
   *
   * @param[in] d_values_in
   *   Pointer to the input sequence of associated value items
   *
   * @param[out] d_values_out
   *   Pointer to the output sequence of associated value items
   *
   * @param[in] num_items
   *   Number of items to be processed
   *
   * @param[in] k
   *   The K value. Will find K elements from num_items elements
   *
   * @param[in] stream
   *   @rst
   *   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
   *   @endrst
   */
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchTopK(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    const KeyInputIteratorT d_keys_in,
    KeyOutputIteratorT d_keys_out,
    const ValueInputIteratorT d_values_in,
    ValueOutputIteratorT d_values_out,
    NumItemsT num_items,
    NumItemsT k,
    cudaStream_t stream,
    int ptx_version)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_keys_in(d_keys_in)
      , d_keys_out(d_keys_out)
      , d_values_in(d_values_in)
      , d_values_out(d_values_out)
      , num_items(num_items)
      , k(k)
      , stream(stream)
      , ptx_version(ptx_version)
  {}

  /******************************************************************************
   * Dispatch entrypoints
   ******************************************************************************/
  template <typename TopKKernelPtrT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE int CalculateBlocksPerSM(TopKKernelPtrT topk_kernel, int block_threads)
  {
    int topk_blocks_per_sm;
    cudaError error;
    do
    {
      error =
        CubDebug(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&topk_blocks_per_sm, topk_kernel, block_threads, 0));
      if (cudaSuccess != error)
      {
        break;
      }
    } while (0);
    return topk_blocks_per_sm;
  }

  template <typename T>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE T RoundUp(T num, T round_value)
  {
    return ((num - 1) / round_value + 1) * round_value;
  }

  template <typename ActivePolicyT,
            typename TopKFirstPassKernelPtrT,
            typename TopKKernelPtrT,
            typename TopKLastFilterKernelPtrT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
  Invoke(TopKFirstPassKernelPtrT topk_firstpass_kernel,
         TopKKernelPtrT topk_kernel,
         TopKLastFilterKernelPtrT topk_lastfilter_kernel)
  {
    using MaxPolicyT = typename SelectedPolicy::MaxPolicy;
    using Policy     = typename ActivePolicyT::TopKPolicyT;

    cudaError error = cudaSuccess;

    constexpr int block_threads    = Policy::BLOCK_THREADS; // Threads per block
    constexpr int items_per_thread = Policy::ITEMS_PER_THREAD; // Items per thread
    constexpr int tile_size        = block_threads * items_per_thread; // Items per block
    int num_tiles                  = static_cast<int>(::cuda::ceil_div(num_items, tile_size)); // Num of blocks
    constexpr int num_passes       = CalcNumPasses<KeyInT, Policy::BITS_PER_PASS>();
    int num_buckets                = 1 << Policy::BITS_PER_PASS;

    do
    {
      // Get device ordinal
      int device_ordinal;
      error = CubDebug(cudaGetDevice(&device_ordinal));
      if (cudaSuccess != error)
      {
        break;
      }

      // Specify temporary storage allocation requirements
      size_t size_counter        = sizeof(Counter<KeyInT, NumItemsT>);
      size_t size_histogram      = num_buckets * sizeof(NumItemsT);
      size_t num_candidates      = cuda::std::max((size_t) 256, (size_t) num_items / Policy::COEFFICIENT_FOR_BUFFER);
      size_t aligend_bytes       = 256;
      size_t allocation_sizes[6] = {
        RoundUp(size_counter, aligend_bytes),
        RoundUp(size_histogram, aligend_bytes),
        RoundUp(num_candidates * sizeof(KeyInT), aligend_bytes),
        RoundUp(num_candidates * sizeof(KeyInT), aligend_bytes),
        KEYS_ONLY ? 0 : RoundUp(num_candidates * sizeof(NumItemsT), aligend_bytes),
        KEYS_ONLY ? 0 : RoundUp(num_candidates * sizeof(NumItemsT), aligend_bytes)};

      // Compute allocation pointers into the single storage blob (or compute the necessary size of the blob)
      void* allocations[6] = {};

      error = CubDebug(detail::AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes));
      if (cudaSuccess != error)
      {
        break;
      }

      if (d_temp_storage == nullptr)
      {
        // Return if the caller is simply requesting the size of the storage allocation
        break;
      }

      // Init the buffer for descriptor and histogram
      error = CubDebug(cudaMemsetAsync(
        allocations[0], 0, static_cast<char*>(allocations[2]) - static_cast<char*>(allocations[0]), stream));
      if (cudaSuccess != error)
      {
        break;
      }

// Log topk_kernel configuration @todo check the kernel launch
#ifdef CUB_DETAIL_DEBUG_ENABLE_LOG
      {
        // Get SM occupancy for select_if_kernel
        if (cudaSuccess != error)
        {
          break;
        }

        _CubLog("Invoking topk_kernel<<<{%d,%d,%d}, %d, 0, "
                "%lld>>>(), %d items per thread, %d SM occupancy\n",
                topk_grid_size.x,
                topk_grid_size.y,
                topk_grid_size.z,
                block_threads,
                (long long) stream,
                items_per_thread,
                topk_blocks_per_sm);
      }
#endif // CUB_DETAIL_DEBUG_ENABLE_LOG

      // Get grid size for scanning tiles
      int device  = -1;
      int num_sms = 0;

      error = CubDebug(cudaGetDevice(&device));
      if (cudaSuccess != error)
      {
        break;
      }
      error = CubDebug(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device));
      if (cudaSuccess != error)
      {
        break;
      }

      dim3 topk_grid_size;
      topk_grid_size.z       = 1;
      topk_grid_size.y       = 1;
      int topk_blocks_per_sm = CalculateBlocksPerSM(topk_kernel, block_threads);
      topk_grid_size.x       = cuda::std::min((unsigned int) topk_blocks_per_sm * num_sms,
                                        (unsigned int) (num_items - 1) / (items_per_thread * block_threads) + 1);
      // Initialize address variables
      Counter<KeyInT, NumItemsT>* counter;
      counter = static_cast<decltype(counter)>(allocations[0]);
      NumItemsT* histogram;
      histogram              = static_cast<decltype(histogram)>(allocations[1]);
      KeyInT* in_buf         = nullptr;
      KeyInT* out_buf        = nullptr;
      NumItemsT* in_idx_buf  = nullptr;
      NumItemsT* out_idx_buf = nullptr;
      int pass               = 0;
      for (; pass < num_passes; pass++)
      {
        // Set operator
        ExtractBinOp<KeyInT, !SelectMin, Policy::BITS_PER_PASS> extract_bin_op(pass);
        IdentifyCandidatesOp<KeyInT, !SelectMin, Policy::BITS_PER_PASS> identify_candidates_op(
          counter->kth_key_bits, pass);

        // Initialize address variables
        in_buf  = static_cast<KeyInT*>(pass % 2 == 0 ? allocations[2] : allocations[3]);
        out_buf = pass == 0 ? nullptr : static_cast<KeyInT*>(pass % 2 == 0 ? allocations[3] : allocations[2]);
        if (!KEYS_ONLY)
        {
          in_idx_buf  = pass <= 1 ? nullptr : static_cast<NumItemsT*>(pass % 2 == 0 ? allocations[4] : allocations[5]);
          out_idx_buf = pass == 0 ? nullptr : static_cast<NumItemsT*>(pass % 2 == 0 ? allocations[5] : allocations[4]);
        }

        // Invoke kernel
        if (pass == 0)
        {
          int topk_blocks_per_sm = CalculateBlocksPerSM(topk_firstpass_kernel, block_threads);
          dim3 topk_firstpass_grid_size;
          topk_firstpass_grid_size.z = 1;
          topk_firstpass_grid_size.y = 1;
          topk_firstpass_grid_size.x =
            cuda::std::min((unsigned int) topk_blocks_per_sm * num_sms,
                           (unsigned int) (num_items - 1) / (items_per_thread * block_threads) + 1);
          THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(topk_firstpass_grid_size, block_threads, 0, stream)
            .doit(
              topk_firstpass_kernel,
              d_keys_in,
              d_keys_out,
              d_values_in,
              d_values_out,
              in_buf,
              in_idx_buf,
              out_buf,
              out_idx_buf,
              counter,
              histogram,
              num_items,
              k,
              extract_bin_op,
              identify_candidates_op,
              pass);
        }
        else
        {
          THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(topk_grid_size, block_threads, 0, stream)
            .doit(
              topk_kernel,
              d_keys_in,
              d_keys_out,
              d_values_in,
              d_values_out,
              in_buf,
              in_idx_buf,
              out_buf,
              out_idx_buf,
              counter,
              histogram,
              num_items,
              k,
              extract_bin_op,
              identify_candidates_op,
              pass);
        }
      }
      // Set operator
      IdentifyCandidatesOp<KeyInT, !SelectMin, Policy::BITS_PER_PASS> identify_candidates_op(
        counter->kth_key_bits, pass);
      topk_blocks_per_sm = CalculateBlocksPerSM(topk_lastfilter_kernel, block_threads);
      topk_grid_size.x   = cuda::std::min((unsigned int) topk_blocks_per_sm * num_sms,
                                        (unsigned int) (num_items - 1) / (items_per_thread * block_threads) + 1);
      THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(topk_grid_size, block_threads, 0, stream)
        .doit(topk_lastfilter_kernel,
              d_keys_in,
              d_keys_out,
              d_values_in,
              d_values_out,
              out_buf,
              out_idx_buf,
              counter,
              histogram,
              num_items,
              k,
              identify_candidates_op,
              pass);
      // pass==num_passes to align with the usage of identify_candidates_op in previous passes.
    } while (0);
    return error;
  }
  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    using MaxPolicyT = typename SelectedPolicy::MaxPolicy;
    return Invoke<ActivePolicyT>(
      DeviceTopKKernel<MaxPolicyT,
                       KeyInputIteratorT,
                       KeyOutputIteratorT,
                       ValueInputIteratorT,
                       ValueOutputIteratorT,
                       NumItemsT,
                       KeyInT,
                       ExtractBinOp<KeyInT, !SelectMin, ActivePolicyT::TopKPolicyT::BITS_PER_PASS>,
                       IdentifyCandidatesOp<KeyInT, !SelectMin, ActivePolicyT::TopKPolicyT::BITS_PER_PASS>,
                       SelectMin,
                       /*IsFirstPass*/ true>,

      DeviceTopKKernel<MaxPolicyT,
                       KeyInputIteratorT,
                       KeyOutputIteratorT,
                       ValueInputIteratorT,
                       ValueOutputIteratorT,
                       NumItemsT,
                       KeyInT,
                       ExtractBinOp<KeyInT, !SelectMin, ActivePolicyT::TopKPolicyT::BITS_PER_PASS>,
                       IdentifyCandidatesOp<KeyInT, !SelectMin, ActivePolicyT::TopKPolicyT::BITS_PER_PASS>,
                       SelectMin,
                       /*IsFirstPass*/ false>,

      DeviceTopKLastFilterKernel<MaxPolicyT,
                                 KeyInputIteratorT,
                                 KeyOutputIteratorT,
                                 ValueInputIteratorT,
                                 ValueOutputIteratorT,
                                 NumItemsT,
                                 KeyInT,
                                 IdentifyCandidatesOp<KeyInT, !SelectMin, ActivePolicyT::TopKPolicyT::BITS_PER_PASS>,
                                 SelectMin>);
  }

  /*
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to `temp_storage_bytes` and no work is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_keys_in
   *   Pointer to the input data of key data to find top K
   *
   * @param[out] d_keys_out
   *   Pointer to the K output sequence of key data
   *
   * @param[in] d_values_in
   *   Pointer to the input sequence of associated value items
   *
   * @param[out] d_values_out
   *   Pointer to the output sequence of associated value items
   *
   * @param[in] num_items
   *   Number of items to be processed
   *
   * @param[in] k
   *   The K value. Will find K elements from num_items elements
   *
   * @param[in] stream
   *   @rst
   *   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
   *   @endrst
   */
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    const KeyInputIteratorT d_keys_in,
    KeyOutputIteratorT d_keys_out,
    const ValueInputIteratorT d_values_in,
    ValueOutputIteratorT d_values_out,
    NumItemsT num_items,
    NumItemsT k,
    cudaStream_t stream)
  {
    using MaxPolicyT = typename SelectedPolicy::MaxPolicy;

    int ptx_version = 0;
    if (cudaError_t error = CubDebug(PtxVersion(ptx_version)))
    {
      return error;
    }

    DispatchTopK dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_keys_out,
      d_values_in,
      d_values_out,
      num_items,
      k,
      stream,
      ptx_version);

    return CubDebug(MaxPolicyT::Invoke(ptx_version, dispatch));
  }
};

CUB_NAMESPACE_END
