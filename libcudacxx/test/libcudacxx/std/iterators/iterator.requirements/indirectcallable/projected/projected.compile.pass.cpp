//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// projected

#include <cuda/std/concepts>
#include <cuda/std/functional>
#include <cuda/std/iterator>

#include "test_iterators.h"

using IntPtr = cuda::std::projected<int const*, cuda::std::identity>;
static_assert(cuda::std::same_as<IntPtr::value_type, int>, "");
static_assert(cuda::std::same_as<decltype(*cuda::std::declval<IntPtr>()), int const&>, "");
static_assert(cuda::std::same_as<cuda::std::iter_difference_t<IntPtr>, cuda::std::ptrdiff_t>, "");

struct S
{};

using Cpp17InputIterator = cuda::std::projected<cpp17_input_iterator<S*>, int S::*>;
static_assert(cuda::std::same_as<Cpp17InputIterator::value_type, int>, "");
static_assert(cuda::std::same_as<decltype(*cuda::std::declval<Cpp17InputIterator>()), int&>, "");
static_assert(cuda::std::same_as<cuda::std::iter_difference_t<Cpp17InputIterator>, cuda::std::ptrdiff_t>, "");

using Cpp20InputIterator = cuda::std::projected<cpp20_input_iterator<S*>, int S::*>;
static_assert(cuda::std::same_as<Cpp20InputIterator::value_type, int>, "");
static_assert(cuda::std::same_as<decltype(*cuda::std::declval<Cpp20InputIterator>()), int&>, "");
static_assert(cuda::std::same_as<cuda::std::iter_difference_t<Cpp20InputIterator>, cuda::std::ptrdiff_t>, "");

using ForwardIterator = cuda::std::projected<forward_iterator<S*>, int (S::*)()>;
static_assert(cuda::std::same_as<ForwardIterator::value_type, int>, "");
static_assert(cuda::std::same_as<decltype(*cuda::std::declval<ForwardIterator>()), int>, "");
static_assert(cuda::std::same_as<cuda::std::iter_difference_t<ForwardIterator>, cuda::std::ptrdiff_t>, "");

using BidirectionalIterator = cuda::std::projected<bidirectional_iterator<S*>, S* (S::*) () const>;
static_assert(cuda::std::same_as<BidirectionalIterator::value_type, S*>, "");
static_assert(cuda::std::same_as<decltype(*cuda::std::declval<BidirectionalIterator>()), S*>, "");
static_assert(cuda::std::same_as<cuda::std::iter_difference_t<BidirectionalIterator>, cuda::std::ptrdiff_t>, "");

using RandomAccessIterator = cuda::std::projected<random_access_iterator<S*>, S && (S::*) ()>;
static_assert(cuda::std::same_as<RandomAccessIterator::value_type, S>, "");
static_assert(cuda::std::same_as<decltype(*cuda::std::declval<RandomAccessIterator>()), S&&>, "");
static_assert(cuda::std::same_as<cuda::std::iter_difference_t<RandomAccessIterator>, cuda::std::ptrdiff_t>, "");

using ContiguousIterator = cuda::std::projected<contiguous_iterator<S*>, S& (S::*) () const>;
static_assert(cuda::std::same_as<ContiguousIterator::value_type, S>, "");
static_assert(cuda::std::same_as<decltype(*cuda::std::declval<ContiguousIterator>()), S&>, "");
static_assert(cuda::std::same_as<cuda::std::iter_difference_t<ContiguousIterator>, cuda::std::ptrdiff_t>, "");

template <class I, class F>
_CCCL_CONCEPT projectable = _CCCL_REQUIRES_EXPR((I, F))(typename(cuda::std::projected<I, F>));

static_assert(!projectable<int, void (*)(int)>, ""); // int isn't indirectly_readable
static_assert(!projectable<S, void (*)(int)>, ""); // S isn't weakly_incrementable
static_assert(!projectable<int*, void(int)>, ""); // void(int) doesn't satisfy indirectly_regular_unary_invcable

int main(int, char**)
{
  return 0;
}
