//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class weekday;

// constexpr bool operator==(const weekday& x, const weekday& y) noexcept;
// constexpr bool operator!=(const weekday& x, const weekday& y) noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_comparisons.h"
#include "test_macros.h"

int main(int, char**)
{
  using weekday = cuda::std::chrono::weekday;

  AssertEqualityAreNoexcept<weekday>();
  AssertEqualityReturnBool<weekday>();

  static_assert(testEqualityValues<weekday>(0U, 0U), "");
  static_assert(testEqualityValues<weekday>(0U, 1U), "");

  //  Some 'ok' values as well
  static_assert(testEqualityValues<weekday>(5U, 5U), "");
  static_assert(testEqualityValues<weekday>(5U, 2U), "");

  for (unsigned i = 0; i < 6; ++i)
  {
    for (unsigned j = 0; j < 6; ++j)
    {
      assert(testEqualityValues<weekday>(i, j));
    }
  }

  return 0;
}
