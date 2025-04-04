//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class month_weekday_last;

// constexpr bool operator==(const month_weekday_last& x, const month_weekday_last& y) noexcept;
//   Returns: x.month() == y.month()
//
// constexpr bool operator< (const month_weekday_last& x, const month_weekday_last& y) noexcept;
//   Returns: x.month() < y.month()

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_comparisons.h"
#include "test_macros.h"

int main(int, char**)
{
  using month              = cuda::std::chrono::month;
  using weekday_last       = cuda::std::chrono::weekday_last;
  using weekday            = cuda::std::chrono::weekday;
  using month_weekday_last = cuda::std::chrono::month_weekday_last;

  constexpr month January     = cuda::std::chrono::January;
  constexpr weekday Tuesday   = cuda::std::chrono::Tuesday;
  constexpr weekday Wednesday = cuda::std::chrono::Wednesday;

  AssertEqualityAreNoexcept<month_weekday_last>();
  AssertEqualityReturnBool<month_weekday_last>();

  static_assert(testEquality(month_weekday_last{cuda::std::chrono::January, weekday_last{Tuesday}},
                             month_weekday_last{cuda::std::chrono::January, weekday_last{Tuesday}},
                             true),
                "");

  static_assert(testEquality(month_weekday_last{cuda::std::chrono::January, weekday_last{Tuesday}},
                             month_weekday_last{cuda::std::chrono::January, weekday_last{Wednesday}},
                             false),
                "");

  //  vary the months
  for (unsigned i = 1; i < 12; ++i)
  {
    for (unsigned j = 1; j < 12; ++j)
    {
      assert((testEquality(month_weekday_last{month{i}, weekday_last{Tuesday}},
                           month_weekday_last{month{j}, weekday_last{Tuesday}},
                           i == j)));
    }
  }

  //  vary the weekday
  for (unsigned i = 0; i < 6; ++i)
  {
    for (unsigned j = 0; j < 6; ++j)
    {
      assert((testEquality(month_weekday_last{January, weekday_last{weekday{i}}},
                           month_weekday_last{January, weekday_last{weekday{j}}},
                           i == j)));
    }
  }

  //  both different
  assert((testEquality(month_weekday_last{month{1}, weekday_last{weekday{1}}},
                       month_weekday_last{month{2}, weekday_last{weekday{2}}},
                       false)));

  return 0;
}
