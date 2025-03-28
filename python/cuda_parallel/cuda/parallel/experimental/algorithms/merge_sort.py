# Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes
from typing import Callable

import numba
from numba.cuda.cudadrv import enums

from .. import _cccl as cccl
from .._bindings import call_build, get_bindings
from .._caching import CachableFunction, cache_with_key
from .._utils import protocols
from ..iterators._iterators import IteratorBase
from ..typing import DeviceArrayLike


def make_cache_key(
    d_in_keys: DeviceArrayLike | IteratorBase,
    d_in_items: DeviceArrayLike | IteratorBase | None,
    d_out_keys: DeviceArrayLike,
    d_out_items: DeviceArrayLike | None,
    op: Callable,
):
    d_in_keys_key = (
        d_in_keys.kind
        if isinstance(d_in_keys, IteratorBase)
        else protocols.get_dtype(d_in_keys)
    )
    if d_in_items is None:
        d_in_items_key = None
    else:
        d_in_items_key = (
            d_in_items.kind
            if isinstance(d_in_items, IteratorBase)
            else protocols.get_dtype(d_in_items)
        )
    d_out_keys_key = protocols.get_dtype(d_out_keys)
    if d_out_items is None:
        d_out_items_key = None
    else:
        d_out_items_key = (
            d_out_items.kind
            if isinstance(d_out_items, IteratorBase)
            else protocols.get_dtype(d_out_items)
        )
    op_key = CachableFunction(op)
    return (d_in_keys_key, d_in_items_key, d_out_keys_key, d_out_items_key, op_key)


class _MergeSort:
    def __init__(
        self,
        d_in_keys: DeviceArrayLike | IteratorBase,
        d_in_items: DeviceArrayLike | IteratorBase | None,
        d_out_keys: DeviceArrayLike,
        d_out_items: DeviceArrayLike | None,
        op: Callable,
    ):
        assert (d_in_items is None) == (d_out_items is None)

        # Referenced from __del__:
        self.build_result = None

        self.d_in_keys_cccl = cccl.to_cccl_iter(d_in_keys)
        self.d_in_items_cccl = cccl.to_cccl_iter(d_in_items)
        self.d_out_keys_cccl = cccl.to_cccl_iter(d_out_keys)
        self.d_out_items_cccl = cccl.to_cccl_iter(d_out_items)

        self.bindings = get_bindings()

        if isinstance(d_in_keys, IteratorBase):
            value_type = d_in_keys.value_type
        else:
            value_type = numba.from_dtype(protocols.get_dtype(d_in_keys))

        sig = (value_type, value_type)
        self.op_wrapper = cccl.to_cccl_op(op, sig)

        self.build_result = cccl.DeviceMergeSortBuildResult()
        error = call_build(
            self.bindings.cccl_device_merge_sort_build,
            ctypes.byref(self.build_result),
            self.d_in_keys_cccl,
            self.d_in_items_cccl,
            self.d_out_keys_cccl,
            self.d_out_items_cccl,
            self.op_wrapper,
        )
        if error != enums.CUDA_SUCCESS:
            raise ValueError("Error building merge_sort")

    def __call__(
        self,
        temp_storage,
        d_in_keys: DeviceArrayLike | IteratorBase,
        d_in_items: DeviceArrayLike | IteratorBase | None,
        d_out_keys: DeviceArrayLike,
        d_out_items: DeviceArrayLike | None,
        num_items: int,
        stream=None,
    ):
        assert (d_in_items is None) == (d_out_items is None)

        set_state_fn = cccl.set_cccl_iterator_state
        set_state_fn(self.d_in_keys_cccl, d_in_keys)
        if d_in_items is not None:
            set_state_fn(self.d_in_items_cccl, d_in_items)
        set_state_fn(self.d_out_keys_cccl, d_out_keys)
        if d_out_items is not None:
            set_state_fn(self.d_out_items_cccl, d_out_items)

        stream_handle = protocols.validate_and_get_stream(stream)
        if temp_storage is None:
            temp_storage_bytes = ctypes.c_size_t()
            d_temp_storage = None
        else:
            temp_storage_bytes = ctypes.c_size_t(temp_storage.nbytes)
            # Note: this is slightly slower, but supports all ndarray-like objects as long as they support CAI
            # TODO: switch to use gpumemoryview once it's ready
            d_temp_storage = temp_storage.__cuda_array_interface__["data"][0]

        error = self.bindings.cccl_device_merge_sort(
            self.build_result,
            ctypes.c_void_p(d_temp_storage),
            ctypes.byref(temp_storage_bytes),
            self.d_in_keys_cccl,
            self.d_in_items_cccl,
            self.d_out_keys_cccl,
            self.d_out_items_cccl,
            ctypes.c_ulonglong(num_items),
            self.op_wrapper,
            ctypes.c_void_p(stream_handle),
        )

        if error != enums.CUDA_SUCCESS:
            raise ValueError("Error in merge sort")

        return temp_storage_bytes.value

    def __del__(self):
        if self.build_result is None:
            return
        self.bindings.cccl_device_merge_sort_cleanup(ctypes.byref(self.build_result))


@cache_with_key(make_cache_key)
def merge_sort(
    d_in_keys: DeviceArrayLike | IteratorBase,
    d_in_items: DeviceArrayLike | IteratorBase | None,
    d_out_keys: DeviceArrayLike,
    d_out_items: DeviceArrayLike | None,
    op: Callable,
):
    """Implements a device-wide merge sort using ``d_in_keys`` and the comparison operator ``op``.

    Example:
        Below, ``merge_sort`` is used to sort a sequence of keys inplace. It also rearranges the items according to the keys' order.

        .. literalinclude:: ../../python/cuda_parallel/tests/test_merge_sort_api.py
          :language: python
          :dedent:
          :start-after: example-begin merge-sort
          :end-before: example-end merge-sort

    Args:
        d_in_keys: Device array or iterator containing the input keys to be sorted
        d_in_items: Optional device array or iterator that contains each key's corresponding item
        d_in_keys: Device array to store the sorted keys
        d_in_items: Device array to store the sorted items
        op: Callable representing the comparison operator

    Returns:
        A callable object that can be used to perform the merge sort
    """
    return _MergeSort(d_in_keys, d_in_items, d_out_keys, d_out_items, op)
