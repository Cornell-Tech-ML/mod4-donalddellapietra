from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides  # noqa: F401

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:  # noqa: D103
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start  # type: ignore[call-overload, assignment]

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        if np.array_equal(out_shape, in_shape) and np.array_equal(
            out_strides, in_strides
        ):
            for i in prange(len(out)):
                out[i] = fn(in_storage[i])

        else:
            for i in prange(len(out)):
                out_index = np.empty(MAX_DIMS, np.int32)
                in_index = np.empty(MAX_DIMS, np.int32)
                to_index(i, out_shape, out_index)

                broadcast_index(out_index, out_shape, in_shape, in_index)

                o = index_to_position(out_index, out_strides)
                j = index_to_position(in_index, in_strides)
                out[o] = fn(in_storage[j])

        # else:

        # # Fast path: when shapes are identical and strides are identical
        # if np.array_equal(out_shape, in_shape) and np.array_equal(out_strides, in_strides):
        #     for i in prange(len(out)):
        #         out[i] = fn(in_storage[i])
        #     return

        # # Regular path: handle broadcasting and different strides
        # for i in prange(len(out)):
        #     to_index(i, out_shape, out_index)
        #     broadcast_index(out_index, out_shape, in_shape, in_index)
        #     o = index_to_position(out_index, out_strides)
        #     j = index_to_position(in_index, in_strides)
        #     out[o] = fn(in_storage[j])

    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        if (
            np.array_equal(out_shape, a_shape)
            and np.array_equal(out_shape, b_shape)
            and np.array_equal(out_strides, a_strides)
            and np.array_equal(out_strides, b_strides)
        ):
            for i in prange(len(out)):
                out[i] = fn(a_storage[i], b_storage[i])

        else:
            for i in prange(len(out)):
                out_idx = np.zeros(MAX_DIMS, np.int32)
                in_idx_a = np.zeros(MAX_DIMS, np.int32)
                in_idx_b = np.zeros(MAX_DIMS, np.int32)
                to_index(i, out_shape, out_idx)

                broadcast_index(out_idx, out_shape, a_shape, in_idx_a)
                broadcast_index(out_idx, out_shape, b_shape, in_idx_b)

                in_pst_a = index_to_position(in_idx_a, a_strides)
                in_pst_b = index_to_position(in_idx_b, b_strides)
                out_pst = index_to_position(out_idx, out_strides)
                out[out_pst] = fn(a_storage[in_pst_a], b_storage[in_pst_b])

    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        # Parallel over all output positions
        for out_index in prange(len(out)):
            # Initialize an index array for the current output position
            out_multi_index = np.empty(MAX_DIMS, np.int32)
            # Get the size of the dimension to be reduced
            reduction_dim_size = a_shape[reduce_dim]

            # Convert the linear index to a multi-dimensional index
            to_index(out_index, out_shape, out_multi_index)
            # Calculate the position in the output storage
            out_storage_pos = index_to_position(out_multi_index, out_strides)

            # Determine the stride for the reduction dimension
            reduction_stride = a_strides[reduce_dim]
            # Calculate the starting position in the input storage
            start_storage_pos = index_to_position(out_multi_index, a_strides)

            # Initialize the accumulator with the current output value
            accumulator = out[out_storage_pos]

            # Iterate over the reduction dimension
            for offset in range(reduction_dim_size):
                # Calculate the current position in the input storage
                current_storage_pos = start_storage_pos + offset * reduction_stride
                # Apply the reduction function
                accumulator = fn(accumulator, float(a_storage[current_storage_pos]))
            # Store the result back in the output storage
            out[out_storage_pos] = accumulator

    return njit(_reduce, parallel=True)  # type: ignore


def _tensor_matrix_multiply(
    out: Storage,
    out_shape: Shape,
    out_strides: Strides,
    a_storage: Storage,
    a_shape: Shape,
    a_strides: Strides,
    b_storage: Storage,
    b_shape: Shape,
    b_strides: Strides,
) -> None:
    """NUMBA tensor matrix multiply function."""
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # Compute dimensions
    batch_size = out_shape[0]
    rows = out_shape[1]
    cols = out_shape[2]
    reduce_size = a_shape[2]

    # Parallelize over batches and rows
    for batch in prange(batch_size):
        for i in prange(rows):
            for j in range(cols):
                # Get output position
                out_pos = (
                    batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]
                )
                # Initialize accumulator
                acc = 0.0

                # Inner reduction loop
                for k in range(reduce_size):
                    # Compute positions in a and b
                    a_pos = batch * a_batch_stride + i * a_strides[1] + k * a_strides[2]
                    b_pos = batch * b_batch_stride + k * b_strides[1] + j * b_strides[2]
                    acc += a_storage[a_pos] * b_storage[b_pos]

                # Store result
                out[out_pos] = acc


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
