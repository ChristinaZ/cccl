//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class year;

// constexpr year operator-(const year& x, const years& y) noexcept;
//   Returns: x + -y.
//
// constexpr years operator-(const year& x, const year& y) noexcept;
//   Returns: If x.ok() == true and y.ok() == true, returns a value m in the range
//   [years{0}, years{11}] satisfying y + m == x.
//   Otherwise the value returned is unspecified.
//   [Example: January - February == years{11}. —end example]

extern "C" int printf(const char*, ...);

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <typename Y, typename Ys>
__host__ __device__ constexpr bool testConstexpr()
{
  Y y{2313};
  Ys offset{1006};
  if (y - offset != Y{1307})
  {
    return false;
  }
  if (y - Y{1307} != offset)
  {
    return false;
  }
  return true;
}

int main(int, char**)
{
  using year  = cuda::std::chrono::year;
  using years = cuda::std::chrono::years;

  static_assert(noexcept(cuda::std::declval<year>() - cuda::std::declval<years>()));
  static_assert(cuda::std::is_same_v<year, decltype(cuda::std::declval<year>() - cuda::std::declval<years>())>);

  static_assert(noexcept(cuda::std::declval<year>() - cuda::std::declval<year>()));
  static_assert(cuda::std::is_same_v<years, decltype(cuda::std::declval<year>() - cuda::std::declval<year>())>);

  static_assert(testConstexpr<year, years>(), "");

  year y{1223};
  for (int i = 1100; i <= 1110; ++i)
  {
    year y1   = y - years{i};
    years ys1 = y - year{i};
    assert(static_cast<int>(y1) == 1223 - i);
    assert(ys1.count() == 1223 - i);
  }

  return 0;
}
