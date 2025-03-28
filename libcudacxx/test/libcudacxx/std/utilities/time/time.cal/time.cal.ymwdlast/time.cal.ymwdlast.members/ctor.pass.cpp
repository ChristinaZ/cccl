//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <chrono>
// class year_month_weekday_last;

//  constexpr year_month_weekday_last(const chrono::year& y, const chrono::month& m,
//                               const chrono::weekday_last& wdl) noexcept;
//
//  Effects:  Constructs an object of type year_month_weekday_last by initializing
//                y_ with y, m_ with m, and wdl_ with wdl.
//
//  constexpr chrono::year                 year() const noexcept;
//  constexpr chrono::month               month() const noexcept;
//  constexpr chrono::weekday           weekday() const noexcept;
//  constexpr chrono::weekday_last weekday_last() const noexcept;
//  constexpr bool                           ok() const noexcept;

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  using year                    = cuda::std::chrono::year;
  using month                   = cuda::std::chrono::month;
  using weekday                 = cuda::std::chrono::weekday;
  using weekday_last            = cuda::std::chrono::weekday_last;
  using year_month_weekday_last = cuda::std::chrono::year_month_weekday_last;

  constexpr month January   = cuda::std::chrono::January;
  constexpr weekday Tuesday = cuda::std::chrono::Tuesday;

  static_assert(noexcept(year_month_weekday_last{year{1}, month{1}, weekday_last{Tuesday}}));

  constexpr year_month_weekday_last ym1{year{2019}, January, weekday_last{Tuesday}};
  static_assert(ym1.year() == year{2019}, "");
  static_assert(ym1.month() == January, "");
  static_assert(ym1.weekday() == Tuesday, "");
  static_assert(ym1.weekday_last() == weekday_last{Tuesday}, "");
  static_assert(ym1.ok(), "");

  return 0;
}
