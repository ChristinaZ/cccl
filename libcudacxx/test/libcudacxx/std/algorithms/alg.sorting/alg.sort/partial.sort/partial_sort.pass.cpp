//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<RandomAccessIterator Iter>
//   requires ShuffleIterator<Iter> && LessThanComparable<Iter::value_type>
//   constexpr void  // constexpr in C++20
//   partial_sort(Iter first, Iter middle, Iter last);

#include <cuda/std/__algorithm_>
#include <cuda/std/cassert>

#include "MoveOnly.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class T, class Iter>
__host__ __device__ constexpr void test()
{
  int orig[15] = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9};
  T work[15]   = {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9};
  for (int n = 0; n < 15; ++n)
  {
    for (int m = 0; m <= n; ++m)
    {
      cuda::std::partial_sort(Iter(work), Iter(work + m), Iter(work + n));
      assert(cuda::std::is_sorted(work, work + m));
      assert(cuda::std::is_permutation(work, work + n, orig));
      // No element in the unsorted portion is less than any element in the sorted portion.
      for (int i = m; i < n; ++i)
      {
        assert(m == 0 || !(work[i] < work[m - 1]));
      }
      cuda::std::copy(orig, orig + 15, work);
    }
  }

  {
    T input[] = {3, 4, 2, 5, 1};
    cuda::std::partial_sort(Iter(input), Iter(input + 3), Iter(input + 5));
    assert(input[0] == 1);
    assert(input[1] == 2);
    assert(input[2] == 3);
    assert(input[3] + input[4] == 4 + 5);
  }
}

__host__ __device__ constexpr bool test()
{
  int i = 42;
  cuda::std::partial_sort(&i, &i, &i); // no-op
  assert(i == 42);

  test<int, random_access_iterator<int*>>();
  test<int, int*>();

  test<MoveOnly, random_access_iterator<MoveOnly*>>();
  test<MoveOnly, MoveOnly*>();

  return true;
}

int main(int, char**)
{
  test();
#if defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
  static_assert(test(), "");
#endif // _CCCL_BUILTIN_IS_CONSTANT_EVALUATED

  return 0;
}
