//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// template<class... B> struct conjunction;                           // C++17
// template<class... B>
//   constexpr bool conjunction_v = conjunction<B...>::value;         // C++17

#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct True
{
  static constexpr bool value = true;
};
struct False
{
  static constexpr bool value = false;
};

int main(int, char**)
{
  static_assert(cuda::std::conjunction<>::value, "");
  static_assert(cuda::std::conjunction<cuda::std::true_type>::value, "");
  static_assert(!cuda::std::conjunction<cuda::std::false_type>::value, "");

  static_assert(cuda::std::conjunction_v<>, "");
  static_assert(cuda::std::conjunction_v<cuda::std::true_type>, "");
  static_assert(!cuda::std::conjunction_v<cuda::std::false_type>, "");

  static_assert(cuda::std::conjunction<cuda::std::true_type, cuda::std::true_type>::value, "");
  static_assert(!cuda::std::conjunction<cuda::std::true_type, cuda::std::false_type>::value, "");
  static_assert(!cuda::std::conjunction<cuda::std::false_type, cuda::std::true_type>::value, "");
  static_assert(!cuda::std::conjunction<cuda::std::false_type, cuda::std::false_type>::value, "");

  static_assert(cuda::std::conjunction_v<cuda::std::true_type, cuda::std::true_type>, "");
  static_assert(!cuda::std::conjunction_v<cuda::std::true_type, cuda::std::false_type>, "");
  static_assert(!cuda::std::conjunction_v<cuda::std::false_type, cuda::std::true_type>, "");
  static_assert(!cuda::std::conjunction_v<cuda::std::false_type, cuda::std::false_type>, "");

  static_assert(cuda::std::conjunction<cuda::std::true_type, cuda::std::true_type, cuda::std::true_type>::value, "");
  static_assert(!cuda::std::conjunction<cuda::std::true_type, cuda::std::false_type, cuda::std::true_type>::value, "");
  static_assert(!cuda::std::conjunction<cuda::std::false_type, cuda::std::true_type, cuda::std::true_type>::value, "");
  static_assert(!cuda::std::conjunction<cuda::std::false_type, cuda::std::false_type, cuda::std::true_type>::value, "");
  static_assert(!cuda::std::conjunction<cuda::std::true_type, cuda::std::true_type, cuda::std::false_type>::value, "");
  static_assert(!cuda::std::conjunction<cuda::std::true_type, cuda::std::false_type, cuda::std::false_type>::value, "");
  static_assert(!cuda::std::conjunction<cuda::std::false_type, cuda::std::true_type, cuda::std::false_type>::value, "");
  static_assert(!cuda::std::conjunction<cuda::std::false_type, cuda::std::false_type, cuda::std::false_type>::value,
                "");

  static_assert(cuda::std::conjunction_v<cuda::std::true_type, cuda::std::true_type, cuda::std::true_type>, "");
  static_assert(!cuda::std::conjunction_v<cuda::std::true_type, cuda::std::false_type, cuda::std::true_type>, "");
  static_assert(!cuda::std::conjunction_v<cuda::std::false_type, cuda::std::true_type, cuda::std::true_type>, "");
  static_assert(!cuda::std::conjunction_v<cuda::std::false_type, cuda::std::false_type, cuda::std::true_type>, "");
  static_assert(!cuda::std::conjunction_v<cuda::std::true_type, cuda::std::true_type, cuda::std::false_type>, "");
  static_assert(!cuda::std::conjunction_v<cuda::std::true_type, cuda::std::false_type, cuda::std::false_type>, "");
  static_assert(!cuda::std::conjunction_v<cuda::std::false_type, cuda::std::true_type, cuda::std::false_type>, "");
  static_assert(!cuda::std::conjunction_v<cuda::std::false_type, cuda::std::false_type, cuda::std::false_type>, "");

  static_assert(cuda::std::conjunction<True>::value, "");
  static_assert(!cuda::std::conjunction<False>::value, "");

  static_assert(cuda::std::conjunction_v<True>, "");
  static_assert(!cuda::std::conjunction_v<False>, "");

  return 0;
}
