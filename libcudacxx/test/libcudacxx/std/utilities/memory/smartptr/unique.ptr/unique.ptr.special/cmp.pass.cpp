//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// <memory>

// unique_ptr

// template <class T1, class D1, class T2, class D2>
//   bool
//   operator==(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y);

// template <class T1, class D1, class T2, class D2>
//   bool
//   operator!=(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y);

// template <class T1, class D1, class T2, class D2>
//   bool
//   operator< (const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y);

// template <class T1, class D1, class T2, class D2>
//   bool
//   operator> (const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y);

// template <class T1, class D1, class T2, class D2>
//   bool
//   operator<=(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y);

// template <class T1, class D1, class T2, class D2>
//   bool
//   operator>=(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y);

// template<class T1, class D1, class T2, class D2>
//   requires three_way_comparable_with<typename unique_ptr<T1, D1>::pointer,
//                                      typename unique_ptr<T2, D2>::pointer>
//   compare_three_way_result_t<typename unique_ptr<T1, D1>::pointer,
//                              typename unique_ptr<T2, D2>::pointer>
//     operator<=>(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y);

#include <cuda/std/__memory_>
#include <cuda/std/cassert>

#include "deleter_types.h"
#include "test_comparisons.h"
#include "test_macros.h"
#include "unique_ptr_test_helper.h"

__host__ __device__ TEST_CONSTEXPR_CXX23 bool test()
{
  AssertComparisonsReturnBool<cuda::std::unique_ptr<int>>();
#if TEST_STD_VER >= 2020 && _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  AssertOrderReturn<cuda::std::strong_ordering, cuda::std::unique_ptr<int>>();
#endif // TEST_STD_VER >= 2020 && _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

  // Pointers of same type
  {
    A* ptr1 = new A;
    A* ptr2 = new A;
    const cuda::std::unique_ptr<A, Deleter<A>> p1(ptr1);
    const cuda::std::unique_ptr<A, Deleter<A>> p2(ptr2);

    assert(!(p1 == p2));
    assert(p1 != p2);
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      assert((p1 < p2) == (ptr1 < ptr2));
      assert((p1 <= p2) == (ptr1 <= ptr2));
      assert((p1 > p2) == (ptr1 > ptr2));
      assert((p1 >= p2) == (ptr1 >= ptr2));
#if TEST_STD_VER >= 2020 && _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
      assert((p1 <=> p2) != cuda::std::strong_ordering::equal);
      assert((p1 <=> p2) == (ptr1 <=> ptr2));
#endif // TEST_STD_VER >= 2020 && _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
    }
  }
  // Pointers of different type
  {
    A* ptr1 = new A;
    B* ptr2 = new B;
    const cuda::std::unique_ptr<A, Deleter<A>> p1(ptr1);
    const cuda::std::unique_ptr<B, Deleter<B>> p2(ptr2);
    assert(!(p1 == p2));
    assert(p1 != p2);
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      assert((p1 < p2) == (ptr1 < ptr2));
      assert((p1 <= p2) == (ptr1 <= ptr2));
      assert((p1 > p2) == (ptr1 > ptr2));
      assert((p1 >= p2) == (ptr1 >= ptr2));
#if TEST_STD_VER >= 2020 && _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
      assert((p1 <=> p2) != cuda::std::strong_ordering::equal);
      assert((p1 <=> p2) == (ptr1 <=> ptr2));
#endif // TEST_STD_VER >= 2020 && _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
    }
  }
  // Pointers of same array type
  {
    A* ptr1 = new A[3];
    A* ptr2 = new A[3];
    const cuda::std::unique_ptr<A[], Deleter<A[]>> p1(ptr1);
    const cuda::std::unique_ptr<A[], Deleter<A[]>> p2(ptr2);
    assert(!(p1 == p2));
    assert(p1 != p2);
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      assert((p1 < p2) == (ptr1 < ptr2));
      assert((p1 <= p2) == (ptr1 <= ptr2));
      assert((p1 > p2) == (ptr1 > ptr2));
      assert((p1 >= p2) == (ptr1 >= ptr2));
#if TEST_STD_VER >= 2020 && _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
      assert((p1 <=> p2) != cuda::std::strong_ordering::equal);
      assert((p1 <=> p2) == (ptr1 <=> ptr2));
#endif // TEST_STD_VER >= 2020 && _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
    }
  }
  // Pointers of different array types
  {
    A* ptr1 = new A[3];
    B* ptr2 = new B[3];
    const cuda::std::unique_ptr<A[], Deleter<A[]>> p1(ptr1);
    const cuda::std::unique_ptr<B[], Deleter<B[]>> p2(ptr2);
    assert(!(p1 == p2));
    assert(p1 != p2);
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      assert((p1 < p2) == (ptr1 < ptr2));
      assert((p1 <= p2) == (ptr1 <= ptr2));
      assert((p1 > p2) == (ptr1 > ptr2));
      assert((p1 >= p2) == (ptr1 >= ptr2));
#if TEST_STD_VER >= 2020 && _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
      assert((p1 <=> p2) != cuda::std::strong_ordering::equal);
      assert((p1 <=> p2) == (ptr1 <=> ptr2));
#endif // TEST_STD_VER >= 2020 && _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
    }
  }
  // Default-constructed pointers of same type
  {
    const cuda::std::unique_ptr<A, Deleter<A>> p1;
    const cuda::std::unique_ptr<A, Deleter<A>> p2;
    assert(p1 == p2);
#if TEST_STD_VER >= 2020 && _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      assert((p1 <=> p2) == cuda::std::strong_ordering::equal);
    }
#endif // TEST_STD_VER >= 2020 && _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  }
  // Default-constructed pointers of different type
  {
    const cuda::std::unique_ptr<A, Deleter<A>> p1;
    const cuda::std::unique_ptr<B, Deleter<B>> p2;
    assert(p1 == p2);
#if TEST_STD_VER >= 2020 && _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
    if (!TEST_IS_CONSTANT_EVALUATED_CXX23())
    {
      assert((p1 <=> p2) == cuda::std::strong_ordering::equal);
    }
#endif // TEST_STD_VER >= 2020 && _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2023
  static_assert(test());
#endif // TEST_STD_VER >= 2023

  return 0;
}
