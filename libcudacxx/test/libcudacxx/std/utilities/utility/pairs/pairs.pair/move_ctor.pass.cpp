//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// pair(pair&&) = default;

#include <cuda/std/utility>
// cuda/std/memory not supported
// #include <cuda/std/memory>
#include <cuda/std/cassert>

#include "test_macros.h"

struct Dummy
{
  Dummy(Dummy const&) = delete;
  Dummy(Dummy&&)      = default;
};

int main(int, char**)
{
  {
    typedef cuda::std::pair<int, short> P1;
    static_assert(cuda::std::is_move_constructible<P1>::value, "");
    P1 p1(3, static_cast<short>(4));
    P1 p2 = cuda::std::move(p1);
    assert(p2.first == 3);
    assert(p2.second == 4);
  }
  {
    using P = cuda::std::pair<Dummy, int>;
    static_assert(!cuda::std::is_copy_constructible<P>::value, "");
    static_assert(cuda::std::is_move_constructible<P>::value, "");
  }

  return 0;
}
