// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * \file
 * cub::AgentTopK implements a stateful abstraction of CUDA thread blocks for participating in device-wide topK.
 */

#pragma once

#include <cub/config.cuh>

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/util_type.cuh>

#include <cuda/atomic>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <iterator>

CUB_NAMESPACE_BEGIN

/**
 * Overload CUDA atomic functions for other 64bit unsigned/signed integer type
 */

using ::atomicAdd;
using ::atomicMin;

_CCCL_DEVICE _CCCL_FORCEINLINE long atomicAdd(long* address, long val)
{
  return (long) atomicAdd((unsigned long long*) address, (unsigned long long) val);
}

_CCCL_DEVICE _CCCL_FORCEINLINE long long atomicAdd(long long* address, long long val)
{
  return (long long) atomicAdd((unsigned long long*) address, (unsigned long long) val);
}

_CCCL_DEVICE _CCCL_FORCEINLINE unsigned long atomicAdd(unsigned long* address, unsigned long val)
{
  return (unsigned long) atomicAdd(reinterpret_cast<unsigned long long*>(address), (unsigned long long) val);
}

_CCCL_DEVICE _CCCL_FORCEINLINE long atomicMin(long* address, long val)
{
  return (long) atomicMin((unsigned long long*) address, (unsigned long long) val);
}

_CCCL_DEVICE _CCCL_FORCEINLINE unsigned long atomicMin(unsigned long* address, unsigned long val)
{
  return (unsigned long) atomicMin(reinterpret_cast<unsigned long long*>(address), (unsigned long long) val);
}

/******************************************************************************
 * Tuning policy types
 ******************************************************************************/
/**
 * Parameterizable tuning policy type for AgentTopK
 *
 * @tparam _BLOCK_THREADS
 *   Threads per thread block
 *
 * @tparam _ITEMS_PER_THREAD
 *   Items per thread (per tile of input)
 *
 * @tparam _BITS_PER_PASS
 *   Number of bits processed per pass
 *
 * @tparam _COEFFICIENT_FOR_BUFFER
 *   The coefficient parameter for reducing the size of buffer.
 *   The size of buffer is `1/ _COEFFICIENT_FOR_BUFFER` of original input
 *
 * @tparam _LOAD_ALGORITHM
 *   The BlockLoad algorithm to use
 *
 * @tparam _SCAN_ALGORITHM
 *   The BlockScan algorithm to use
 */

template <int _BLOCK_THREADS,
          int _ITEMS_PER_THREAD,
          int _BITS_PER_PASS,
          int _COEFFICIENT_FOR_BUFFER,
          BlockLoadAlgorithm _LOAD_ALGORITHM,
          BlockScanAlgorithm _SCAN_ALGORITHM>
struct AgentTopKPolicy
{
  /// Threads per thread block
  static constexpr int BLOCK_THREADS = _BLOCK_THREADS;
  /// Items per thread (per tile of input)
  static constexpr int ITEMS_PER_THREAD = _ITEMS_PER_THREAD;
  /// Number of BITS Processed per pass
  static constexpr int BITS_PER_PASS = _BITS_PER_PASS;
  /// Coefficient for reducing buffer size
  static constexpr int COEFFICIENT_FOR_BUFFER = _COEFFICIENT_FOR_BUFFER;

  /// The BlockLoad algorithm to use
  static constexpr BlockLoadAlgorithm LOAD_ALGORITHM = _LOAD_ALGORITHM;

  /// The BlockScan algorithm to use
  static constexpr BlockScanAlgorithm SCAN_ALGORITHM = _SCAN_ALGORITHM;
};

/******************************************************************************
 * Thread block abstractions
 ******************************************************************************/

template <typename KeyInT, typename NumItemsT>
struct alignas(128) Counter
{
  // We are processing the values in multiple passes, from most significant to least
  // significant. In each pass, we keep the length of input (`len`) and the `k` of
  // current pass, and update them at the end of the pass.
  NumItemsT k;
  NumItemsT len;

  //  `previous_len` is the length of input in previous pass. Note that `previous_len`
  //  rather than `len` is used for the filtering step because filtering is indeed for
  //  previous pass.
  NumItemsT previous_len;

  // We determine the bits of the k_th key inside the mask processed by the pass. The
  // already known bits are stored in `kth_key_bits`. It's used to discriminate a
  // element is a result (written to `out`), a candidate for next pass (written to
  // `out_buf`), or not useful (discarded). The bits that are not yet processed do not
  // matter for this purpose.
  typename Traits<KeyInT>::UnsignedBits kth_key_bits;

  // Record how many elements have passed filtering. It's used to determine the position
  // in the `out_buf` where an element should be written.
  alignas(128) NumItemsT filter_cnt;

  // For a row inside a batch, we may launch multiple thread blocks. This counter is
  // used to determine if the current block is the last running block. If so, this block
  // will execute Scan() and ChooseBucket().
  alignas(128) unsigned int finished_block_cnt;

  // Record how many elements have been written to the front of `out`. Elements less (if
  // SELECT_MIN==true) than the k-th key are written from front to back.
  alignas(128) NumItemsT out_cnt;

  // Record how many elements have been written to the back of `out`. Elements equal to
  // the k-th key are written from back to front. We need to keep count of them
  // separately because the number of elements that <= the k-th key might exceed k.
  alignas(128) NumItemsT out_back_cnt;
  // The 'alignas' is necessary to improve the performance of global memory accessing by isolating the request,
  // especially for the segment version.
};

/**
 * Operations for calculating the bin index based on the input
 */
template <typename T, int BITS_PER_PASS>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr int CalcNumPasses()
{
  return ::cuda::ceil_div<int>(sizeof(T) * 8, BITS_PER_PASS);
}

/**
 * Bit 0 is the least significant (rightmost);
 * this implementation processes input from the most to the least significant bit.
 * This way, we can skip some passes in the end at the cost of having an unsorted output.
 *
 */
template <typename T, int BITS_PER_PASS>
_CCCL_DEVICE constexpr int CalcStartBit(const int pass)
{
  int start_bit = static_cast<int>(sizeof(T) * 8) - (pass + 1) * BITS_PER_PASS;
  if (start_bit < 0)
  {
    start_bit = 0;
  }
  return start_bit;
}

/**
 * Used in the bin ID calculation to exclude bits unrelated to the current pass
 */
template <typename T, int BITS_PER_PASS>
_CCCL_DEVICE constexpr unsigned CalcMask(const int pass)
{
  int num_bits = CalcStartBit<T, BITS_PER_PASS>(pass - 1) - CalcStartBit<T, BITS_PER_PASS>(pass);
  return (1 << num_bits) - 1;
}

/**
 * Get the bin ID from the value of element
 */
template <typename T, bool FLIP, int BITS_PER_PASS>
struct ExtractBinOp
{
  int pass{};
  int start_bit;
  unsigned mask;

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE ExtractBinOp(int pass)
      : pass(pass)
  {
    start_bit = CalcStartBit<T, BITS_PER_PASS>(pass);
    mask      = CalcMask<T, BITS_PER_PASS>(pass);
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int operator()(T key) const
  {
    auto bits = reinterpret_cast<typename Traits<T>::UnsignedBits&>(key);
    bits      = Traits<T>::TwiddleIn(bits);

    if constexpr (FLIP)
    {
      bits = ~bits;
    }
    int bucket = (bits >> start_bit) & mask;
    return bucket;
  }
};

/**
 * Check if the input element is still a candidate for the target pass.
 */
template <typename T, bool FLIP, int BITS_PER_PASS>
struct IdentifyCandidatesOp
{
  typename Traits<T>::UnsignedBits& kth_key_bits;
  int pass;
  int start_bit;
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE IdentifyCandidatesOp(typename Traits<T>::UnsignedBits& kth_key_bits, int pass)
      : kth_key_bits(kth_key_bits)
      , pass(pass - 1)
  {
    start_bit = CalcStartBit<T, BITS_PER_PASS>(this->pass);
  }

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE int operator()(T key) const
  {
    auto bits = reinterpret_cast<typename Traits<T>::UnsignedBits&>(key);
    bits      = Traits<T>::TwiddleIn(bits);

    if constexpr (FLIP)
    {
      bits = ~bits;
    }

    bits = (bits >> start_bit) << start_bit;

    return (bits < kth_key_bits) ? -1 : (bits == kth_key_bits) ? 0 : 1;
  }
};

/**
 * @brief AgentTopK implements a stateful abstraction of CUDA thread blocks for participating in
 * device-wide topK
 *
 * @tparam AgentTopKPolicyT
 *   Parameterized AgentTopKPolicy tuning policy type
 *
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
 * @tparam ExtractBinOpT
 *   Operations to extract the bin from the input key values
 *
 * @tparam IdentifyCandidatesOpT
 *   Operations to filter the input key values
 *
 * @tparam NumItemsT
 *   Type of variable num_items and k
 *
 * @tparam SELECT_MIN
 *   Determine whether to select the smallest (SELECT_MIN=true) or largest (SELECT_MIN=false) K elements.
 *
 */

template <typename AgentTopKPolicyT,
          typename KeyInputIteratorT,
          typename KeyOutputIteratorT,
          typename ValueInputIteratorT,
          typename ValueOutputIteratorT,
          typename ExtractBinOpT,
          typename IdentifyCandidatesOpT,
          typename NumItemsT,
          bool SELECT_MIN,
          bool IS_DETERMINISTIC>
struct AgentTopK
{
  //---------------------------------------------------------------------
  // Types and constants
  //---------------------------------------------------------------------
  // The key and value type
  using KeyInT = detail::it_value_t<KeyInputIteratorT>;

  static constexpr int BLOCK_THREADS          = AgentTopKPolicyT::BLOCK_THREADS;
  static constexpr int ITEMS_PER_THREAD       = AgentTopKPolicyT::ITEMS_PER_THREAD;
  static constexpr int BITS_PER_PASS          = AgentTopKPolicyT::BITS_PER_PASS;
  static constexpr int COEFFICIENT_FOR_BUFFER = AgentTopKPolicyT::COEFFICIENT_FOR_BUFFER;
  static constexpr int TILE_ITEMS             = BLOCK_THREADS * ITEMS_PER_THREAD;
  static constexpr int num_buckets            = 1 << BITS_PER_PASS;

  static constexpr bool KEYS_ONLY                = std::is_same<ValueInputIteratorT, NullType>::value;
  static constexpr int items_per_thread_for_scan = (num_buckets - 1) / BLOCK_THREADS + 1;

  // Parameterized BlockLoad type for input data
  using BlockLoadInputT = BlockLoad<KeyInT, BLOCK_THREADS, ITEMS_PER_THREAD, AgentTopKPolicyT::LOAD_ALGORITHM>;
  using BlockLoadTransT = BlockLoad<NumItemsT, BLOCK_THREADS, items_per_thread_for_scan, BLOCK_LOAD_TRANSPOSE>;
  // Parameterized BlockScan type
  using BlockScanT = BlockScan<NumItemsT, BLOCK_THREADS, AgentTopKPolicyT::SCAN_ALGORITHM>;
  // Parameterized BlockStore type
  using BlockStoreTransT = BlockStore<NumItemsT, BLOCK_THREADS, items_per_thread_for_scan, BLOCK_STORE_TRANSPOSE>;

  // Shared memory
  union _TempStorage
  {
    // Smem needed for loading
    typename BlockLoadInputT::TempStorage load_input;
    typename BlockLoadTransT::TempStorage load_trans;
    // Smem needed for scan
    typename BlockScanT::TempStorage scan;
    // Smem needed for storing
    typename BlockStoreTransT::TempStorage store_trans;
  };
  /// Alias wrapper allowing storage to be unioned
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  //---------------------------------------------------------------------
  // Per-thread fields
  //---------------------------------------------------------------------
  _TempStorage& temp_storage; ///< Reference to temp_storage
  KeyInputIteratorT d_keys_in; ///< Input keys
  KeyOutputIteratorT d_keys_out; ///< Output keys
  ValueInputIteratorT d_values_in; ///< Input values
  ValueOutputIteratorT d_values_out; ///< Output values
  NumItemsT num_items; ///< Total number of input items
  NumItemsT k; ///< Total number of output items
  ExtractBinOpT extract_bin_op; /// The operation for bin
  IdentifyCandidatesOpT identify_candidates_op; /// The operation for filtering
  bool load_from_original_input; /// Set if loading data from original input

  //---------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------
  /**
   * @param temp_storage
   *   Reference to temp_storage
   *
   * @param d_keys_in
   *   Input data, keys
   *
   * @param d_keys_out
   *   Output data, keys
   *
   * @param d_values_in
   *   Input data, values
   *
   * @param d_values_out
   *   Output data, values
   *
   * @param num_items
   *   Total number of input items
   *
   * @param k
   *   The K value. Will find K elements from num_items elements
   *
   * @param extract_bin_op
   *   Extract bin operator
   *
   * @param identify_candidates_op
   *   Filter operator
   *
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE AgentTopK(
    TempStorage& temp_storage,
    const KeyInputIteratorT d_keys_in,
    KeyOutputIteratorT d_keys_out,
    const ValueInputIteratorT d_values_in,
    ValueOutputIteratorT d_values_out,
    NumItemsT num_items,
    NumItemsT k,
    ExtractBinOpT extract_bin_op,
    IdentifyCandidatesOpT identify_candidates_op)
      : temp_storage(temp_storage.Alias())
      , d_keys_in(d_keys_in)
      , d_keys_out(d_keys_out)
      , d_values_in(d_values_in)
      , d_values_out(d_values_out)
      , num_items(num_items)
      , k(k)
      , extract_bin_op(extract_bin_op)
      , identify_candidates_op(identify_candidates_op)
  {}

  //---------------------------------------------------------------------
  // Utility methods for device topK
  //---------------------------------------------------------------------
  /**
   * Try to perform vectorized data loading.
   */
  template <typename Func, typename T>
  _CCCL_DEVICE _CCCL_FORCEINLINE void ConsumeRange(T in, const NumItemsT num_items, Func f)
  {
    KeyInT thread_data[ITEMS_PER_THREAD];

    const NumItemsT ITEMS_PER_PASS   = TILE_ITEMS * gridDim.x;
    const NumItemsT total_num_blocks = (num_items - 1) / TILE_ITEMS + 1;

    const NumItemsT num_remaining_elements = num_items % TILE_ITEMS;
    const NumItemsT last_block_id          = (total_num_blocks - 1) % gridDim.x;

    NumItemsT tile_base = blockIdx.x * TILE_ITEMS;
    NumItemsT offset    = threadIdx.x * ITEMS_PER_THREAD + tile_base;

    for (int i_block = blockIdx.x; i_block < total_num_blocks - 1; i_block += gridDim.x)
    {
      BlockLoadInputT(temp_storage.load_input).Load(in + tile_base, thread_data);
      for (int j = 0; j < ITEMS_PER_THREAD; ++j)
      {
        f(thread_data[j], offset + j);
      }
      tile_base += ITEMS_PER_PASS;
      offset += ITEMS_PER_PASS;
    }

    if (blockIdx.x == last_block_id)
    {
      if (num_remaining_elements == 0)
      {
        BlockLoadInputT(temp_storage.load_input).Load(in + tile_base, thread_data);
      }
      else
      {
        BlockLoadInputT(temp_storage.load_input).Load(in + tile_base, thread_data, num_remaining_elements, 0);
      }

      for (int j = 0; j < ITEMS_PER_THREAD; ++j)
      {
        if ((offset + j) < num_items)
        {
          f(thread_data[j], offset + j);
        }
      }
    }
  }
  /**
   * Fused filtering of the current pass and building histogram for the next pass
   */
  template <bool IS_FIRST_PASS>
  _CCCL_DEVICE _CCCL_FORCEINLINE void FilterAndHistogram(
    KeyInT* in_buf,
    NumItemsT* in_idx_buf,
    KeyInT* out_buf,
    NumItemsT* out_idx_buf,
    NumItemsT previous_len,
    Counter<KeyInT, NumItemsT>* counter,
    NumItemsT* histogram,
    NumItemsT* histogram_smem,
    int pass,
    bool early_stop)
  {
    if constexpr (IS_FIRST_PASS)
    {
      // Passed to vectorized loading, this function executes in all blocks in parallel,
      // i.e. the work is split along the input (both, in batches and chunks of a single
      // row). Later, the histograms are merged using atomicAdd.
      auto f = [this, histogram_smem](KeyInT key, NumItemsT index) {
        int bucket = extract_bin_op(key);
        atomicAdd(histogram_smem + bucket, static_cast<NumItemsT>(1));
      };
      ConsumeRange(d_keys_in, previous_len, f);
    }
    else
    {
      NumItemsT* p_filter_cnt = &counter->filter_cnt;
      NumItemsT* p_out_cnt    = &counter->out_cnt;
      const auto kth_key_bits = counter->kth_key_bits;

      // See the remark above on the distributed execution of `f` using
      // vectorized loading.
      auto f =
        [in_idx_buf,
         out_buf,
         out_idx_buf,
         kth_key_bits,
         counter,
         p_filter_cnt,
         p_out_cnt,
         this,
         histogram_smem,
         early_stop,
         pass](KeyInT key, NumItemsT i) {
          int pre_res = identify_candidates_op(key);
          if (pre_res == 0)
          {
            NumItemsT index;
            NumItemsT pos;
            if (early_stop)
            {
              pos             = atomicAdd(p_out_cnt, static_cast<NumItemsT>(1));
              d_keys_out[pos] = key;

              if constexpr (!KEYS_ONLY)
              {
                index             = in_idx_buf ? in_idx_buf[i] : i;
                d_values_out[pos] = d_values_in[index];
              }
            }
            else
            {
              if (out_buf)
              {
                pos          = atomicAdd(p_filter_cnt, static_cast<NumItemsT>(1));
                out_buf[pos] = key;
                if constexpr (!KEYS_ONLY)
                {
                  index            = in_idx_buf ? in_idx_buf[i] : i;
                  out_idx_buf[pos] = index;
                }
              }

              int bucket = extract_bin_op(key);
              atomicAdd(histogram_smem + bucket, static_cast<NumItemsT>(1));
            }
          }
          // the condition `(out_buf || early_stop)` is a little tricky:
          // If we skip writing to `out_buf` (when `out_buf` is nullptr), we should skip
          // writing to `out` too. So we won't write the same key to `out` multiple
          // times in different passes. And if we keep skipping the writing, keys will
          // be written in `LastFilter_kernel()` at last. But when `early_stop` is
          // true, we need to write to `out` since it's the last chance.
          else if ((out_buf || early_stop) && (pre_res < 0))
          {
            NumItemsT pos   = atomicAdd(p_out_cnt, static_cast<NumItemsT>(1));
            d_keys_out[pos] = key;
            if constexpr (!KEYS_ONLY)
            {
              NumItemsT index   = in_idx_buf ? in_idx_buf[i] : i;
              d_values_out[pos] = d_values_in[index];
            }
          }
        };
      if (load_from_original_input)
      {
        ConsumeRange(d_keys_in, previous_len, f);
      }
      else
      {
        ConsumeRange(in_buf, previous_len, f);
      }
    }

    if (early_stop)
    {
      return;
    }
    __syncthreads();

    // merge histograms produced by individual blocks
    for (int i = threadIdx.x; i < num_buckets; i += blockDim.x)
    {
      if (histogram_smem[i] != 0)
      {
        atomicAdd(histogram + i, histogram_smem[i]);
      }
    }
  }

  /**
   * Replace histogram with its own prefix sum
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void Scan(volatile NumItemsT* histogram, NumItemsT* histogram_smem)
  {
    NumItemsT thread_data[items_per_thread_for_scan];

    BlockLoadTransT(temp_storage.load_trans).Load(histogram, thread_data, num_buckets, 0);
    __syncthreads();

    BlockScanT(temp_storage.scan).InclusiveSum(thread_data, thread_data);
    __syncthreads();

    BlockStoreTransT(temp_storage.store_trans).Store(histogram_smem, thread_data, num_buckets);
  }

  /**
   * Calculate in which bucket the k-th value will fall
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  ChooseBucket(Counter<KeyInT, NumItemsT>* counter, const NumItemsT* histogram, const NumItemsT k, const int pass)
  {
    for (int i = threadIdx.x; i < num_buckets; i += blockDim.x)
    {
      NumItemsT prev = (i == 0) ? 0 : histogram[i - 1];
      NumItemsT cur  = histogram[i];

      // one and only one thread will satisfy this condition, so counter is written by
      // only one thread
      if (prev < k && cur >= k)
      {
        counter->k                                   = k - prev; // how many values still are there to find
        counter->len                                 = cur - prev; // number of values in next pass
        typename Traits<KeyInT>::UnsignedBits bucket = i;
        int start_bit                                = CalcStartBit<KeyInT, BITS_PER_PASS>(pass);
        counter->kth_key_bits |= bucket << start_bit;
      }
    }
  }

  /**
   * @brief Perform the filter operation for the lass pass.
   *
   * @param in_buf
   *   Buffer address for input data
   *
   * @param in_idx_buf
   *   Buffer address for index of the input data
   *
   * @param counter
   *   Record the meta data for different passes
   *
   * @param histogram
   *   Record the element number of each bucket
   *
   * @param k
   *   The original K value. Will find K elements from num_items elements
   *
   * @param pass
   *   Indicate which pass are processed currently
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE void InvokeLastFilter(
    KeyInT* in_buf,
    NumItemsT* in_idx_buf,
    NumItemsT* out_idx_buf,
    Counter<KeyInT, NumItemsT>* counter,
    NumItemsT* histogram,
    int k,
    int pass)
  {
    const NumItemsT buf_len = cuda::std::max((NumItemsT) 256, num_items / COEFFICIENT_FOR_BUFFER);

    load_from_original_input = counter->previous_len > buf_len;
    NumItemsT current_len    = load_from_original_input ? num_items : counter->previous_len;
    in_idx_buf               = load_from_original_input ? nullptr : in_idx_buf;

    if (current_len == 0)
    {
      return;
    }

    // changed in ChooseBucket(); need to reload
    NumItemsT num_of_kth_needed = counter->k;
    NumItemsT* p_out_cnt        = &counter->out_cnt;
    NumItemsT* p_out_back_cnt   = &counter->out_back_cnt;

    NumItemsT* p_equal = out_idx_buf + k - num_of_kth_needed;
    cuda::atomic_ref<NumItemsT> ref_last(p_equal[num_of_kth_needed - 1]);

    auto f =
      [this, pass, p_out_cnt, counter, in_idx_buf, p_out_back_cnt, num_of_kth_needed, k, current_len, ref_last, p_equal](
        KeyInT key, NumItemsT i) {
        int res = identify_candidates_op(key);
        if (res < 0)
        {
          NumItemsT pos   = atomicAdd(p_out_cnt, static_cast<NumItemsT>(1));
          d_keys_out[pos] = key;
          if constexpr (!KEYS_ONLY)
          {
            NumItemsT index = in_idx_buf ? in_idx_buf[i] : i;

            // For one-block version, `in_idx_buf` could be nullptr at pass 0.
            // For non one-block version, if writing has been skipped, `in_idx_buf` could
            // be nullptr if `in_buf` is `in`
            d_values_out[pos] = d_values_in[index];
          }
        }
        else if (res == 0)
        {
          NumItemsT new_idx  = in_idx_buf ? in_idx_buf[i] : i;
          NumItemsT back_pos = atomicAdd(p_out_back_cnt, static_cast<NumItemsT>(1));

          if (back_pos < num_of_kth_needed)
          {
            NumItemsT pos   = k - 1 - back_pos;
            d_keys_out[pos] = key;

            if constexpr (!KEYS_ONLY && (!IS_DETERMINISTIC))
            {
              d_values_out[pos] = d_values_in[new_idx];
            }
          }
          if constexpr (!KEYS_ONLY && IS_DETERMINISTIC)
          {
            if (new_idx < ref_last.load(cuda::memory_order_relaxed))
            {
              for (int j = 0; j < num_of_kth_needed; j++)
              {
                NumItemsT pre_idx = atomicMin(&p_equal[j], new_idx);
                if (pre_idx > new_idx)
                {
                  new_idx = pre_idx;
                }
              }
            }
          }
        }
      };

    if (load_from_original_input)
    {
      ConsumeRange(d_keys_in, current_len, f);
    }
    else
    {
      ConsumeRange(in_buf, current_len, f);
    }

    __threadfence();
    __syncthreads();

    bool is_last_block = false;
    if (threadIdx.x == 0)
    {
      unsigned int finished = atomicInc(&counter->finished_block_cnt, gridDim.x - 1);
      is_last_block         = (finished == (gridDim.x - 1));
    }

    if (__syncthreads_or(is_last_block))
    {
      // Store the final value
      if constexpr (!KEYS_ONLY && IS_DETERMINISTIC)
      {
        for (int j = threadIdx.x; j < num_of_kth_needed; j += static_cast<size_t>(blockDim.x))
        {
          NumItemsT index                         = p_equal[j];
          d_values_out[k - num_of_kth_needed + j] = d_values_in[index];
        }
      }
    }
  }

  /**
   * @brief Perform the histogram collection, prefix sum calculation and candidates filter (except the filtering in the
   * last pas)
   *
   * @param in_buf
   *   Buffer address for input data
   *
   * @param in_idx_buf
   *   Buffer address for index of the input data
   *
   * @param out_buf
   *   Buffer address for output data
   *
   * @param out_idx_buf
   *   Buffer address for index of the output data
   *
   * @param counter
   *   Record the meta data for different passes
   *
   * @param histogram
   *   Record the element number of each bucket
   *
   * @param pass
   *   Indicate which pass are processed currently
   */
  template <bool IS_FIRST_PASS>
  _CCCL_DEVICE _CCCL_FORCEINLINE void InvokeOneSweep(
    KeyInT* in_buf,
    NumItemsT* in_idx_buf,
    KeyInT* out_buf,
    NumItemsT* out_idx_buf,
    NumItemsT* lastpass_idx_buf,
    Counter<KeyInT, NumItemsT>* counter,
    NumItemsT* histogram,
    int pass)
  {
    NumItemsT current_k;
    NumItemsT previous_len;
    NumItemsT current_len;

    if (pass == 0)
    {
      current_k    = k;
      previous_len = num_items;
      current_len  = num_items;
    }
    else
    {
      current_k    = counter->k;
      current_len  = counter->len;
      previous_len = counter->previous_len;
    }

    if (current_len == 0)
    {
      return;
    }

    const bool early_stop   = (current_len == current_k);
    const NumItemsT buf_len = cuda::std::max((NumItemsT) 256, num_items / COEFFICIENT_FOR_BUFFER);
    if (previous_len > buf_len)
    {
      load_from_original_input = true;
      in_idx_buf               = nullptr;
      previous_len             = num_items;
    }
    else
    {
      load_from_original_input = false;
    }

    // "current_len > buf_len" means current pass will skip writing buffer
    if (current_len > buf_len)
    {
      out_buf     = nullptr;
      out_idx_buf = nullptr;
    }

    __shared__ NumItemsT histogram_smem[num_buckets];
    for (NumItemsT i = threadIdx.x; i < num_buckets; i += blockDim.x)
    {
      histogram_smem[i] = 0;
    }
    __syncthreads();

    FilterAndHistogram<IS_FIRST_PASS>(
      in_buf, in_idx_buf, out_buf, out_idx_buf, previous_len, counter, histogram, histogram_smem, pass, early_stop);

    __threadfence();
    // We need this `__threadfence()` because the global array `histogram` will be accessed by other threads with
    // non-atomic operations.

    bool is_last_block = false;
    if (threadIdx.x == 0)
    {
      unsigned int finished = atomicInc(&counter->finished_block_cnt, gridDim.x - 1);
      is_last_block         = (finished == (gridDim.x - 1));
    }

    if (__syncthreads_or(is_last_block))
    {
      if (threadIdx.x == 0)
      {
        if (early_stop)
        {
          // `LastFilter_kernel()` requires setting previous_len
          counter->previous_len = 0;
          counter->len          = 0;
        }
        else
        {
          counter->previous_len = current_len;
          // not necessary for the last pass, but put it here anyway
          counter->filter_cnt = 0;
        }
      }

      constexpr int num_passes = CalcNumPasses<KeyInT, BITS_PER_PASS>();
      Scan(histogram, histogram_smem);
      __syncthreads();
      ChooseBucket(counter, histogram_smem, current_k, pass);

      // reset for next pass
      if (pass != num_passes - 1)
      {
        for (int i = threadIdx.x; i < num_buckets; i += blockDim.x)
        {
          histogram[i] = 0;
        }
      }

      if (pass == num_passes - 1)
      {
        if constexpr (!KEYS_ONLY && IS_DETERMINISTIC)
        {
          volatile const NumItemsT num_of_kth_needed = counter->k;
          for (NumItemsT i = threadIdx.x; i < num_of_kth_needed; i += blockDim.x)
          {
            lastpass_idx_buf[k - num_of_kth_needed + i] = cuda::std::numeric_limits<NumItemsT>::max();
          }
        }
      }
    }
  }
};

CUB_NAMESPACE_END
