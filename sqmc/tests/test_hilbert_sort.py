"""Tests for the Hilbert index and sorting functions in ``sqmc.hilbert_sort``.

Test conventions follow ``state-space-models/cuthbert``: ``chex.TestCase``
classes, ``@chex.variants(with_jit, without_jit)`` for pure array-valued
transformations, absl ``parameterized`` markers, module-level
``chex.assert_trees_all_close`` and ``chex.assert_shape``, and a
module-autouse x64 fixture that restores the flag on teardown.
"""

import sys
from pathlib import Path

import jax

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import parameterized

import chex

from sqmc.hilbert_sort.hilbert_sort import (
    Hilbert_to_int,
    gray_decode,
    gray_decode_travel,
    gray_encode,
    gray_encode_travel,
    hilbert_sort,
)


@pytest.fixture(scope="module", autouse=True)
def config():
    """Enable double precision for the module and restore it on teardown."""
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


class GrayCodeTest(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    def test_round_trip(self):
        values = jnp.asarray(
            [0, 1, 2, 3, 17, 2**31, 2**62 - 1],
            dtype=jnp.uint64,
        )
        chex.assert_trees_all_close(
            self.variant(lambda: gray_decode(gray_encode(values)))(),
            values,
            rtol=0.0,
            atol=0.0,
        )

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.parameters([2, 7, 31, 62])
    def test_traveling_gray_code_round_trip_at_wide_dimensions(self, dimension):
        mask = jnp.uint64((1 << dimension) - 1)
        start = jnp.uint64((1 << (dimension - 1)) - 1)
        end = start ^ jnp.uint64(1 << (dimension - 1))
        steps = jnp.asarray(
            [0, 1, 2, 3, (1 << dimension) - 1],
            dtype=jnp.uint64,
        )

        def encode():
            return jax.vmap(
                lambda step: gray_encode_travel(start, end, mask, step)
            )(steps)

        def decode(encoded):
            return jax.vmap(
                lambda value: gray_decode_travel(start, end, mask, value)
            )(encoded)

        decoded = self.variant(decode)(self.variant(encode)())
        chex.assert_trees_all_close(decoded, steps, rtol=0.0, atol=0.0)


class HilbertIndexTest(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    def test_known_two_dimensional_indices(self):
        coordinates = jnp.asarray(
            [
                [0, 0],
                [0, 1],
                [0, 2],
                [0, 3],
                [1, 0],
                [1, 1],
                [1, 2],
                [1, 3],
                [2, 0],
                [2, 1],
                [2, 2],
                [2, 3],
                [3, 0],
                [3, 1],
                [3, 2],
                [3, 3],
            ],
            dtype=jnp.uint64,
        )
        expected = jnp.asarray(
            [0, 3, 4, 5, 1, 2, 7, 6, 14, 13, 8, 9, 15, 12, 11, 10],
            dtype=jnp.uint64,
        )
        actual = self.variant(
            lambda: jax.vmap(lambda point: Hilbert_to_int(point, 4))(coordinates)
        )()
        chex.assert_trees_all_close(actual, expected, rtol=0.0, atol=0.0)


class HilbertSortTest(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    def test_one_dimensional_sort_matches_argsort(self):
        values = jnp.asarray([3.0, 1.0, 4.0, 1.0, 5.0])
        expected = jnp.argsort(values, stable=True)

        chex.assert_trees_all_close(
            self.variant(lambda: hilbert_sort(values))(),
            expected,
            rtol=0.0,
            atol=0.0,
        )
        chex.assert_trees_all_close(
            self.variant(lambda: hilbert_sort(values[:, None]))(),
            expected,
            rtol=0.0,
            atol=0.0,
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_constant_columns_are_handled_without_nan_dependent_ordering(self):
        points = jnp.ones((16, 3), dtype=jnp.float64)
        chex.assert_trees_all_close(
            self.variant(lambda: hilbert_sort(points))(),
            jnp.arange(16),
            rtol=0.0,
            atol=0.0,
        )

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.parameters([2, 4, 7, 31, 62])
    def test_sort_returns_a_permutation_for_supported_dimensions(self, dimension):
        points = jax.random.normal(
            jax.random.PRNGKey(dimension),
            shape=(128, dimension),
            dtype=jnp.float64,
        )
        order = self.variant(lambda: hilbert_sort(points))()
        chex.assert_trees_all_close(
            jnp.sort(order),
            jnp.arange(points.shape[0]),
            rtol=0.0,
            atol=0.0,
        )

    def test_empty_input_returns_empty_permutation(self):
        order = hilbert_sort(jnp.empty((0, 2), dtype=jnp.float64))
        chex.assert_shape(order, (0,))

    def test_more_than_62_dimensions_is_rejected(self):
        with pytest.raises(ValueError, match="At most 62 dimensions"):
            hilbert_sort(jnp.ones((4, 63), dtype=jnp.float64))

    @parameterized.parameters([((2, 2, 2),), ((2, 0),)])
    def test_invalid_shape_is_rejected(self, shape):
        with pytest.raises(ValueError):
            hilbert_sort(jnp.ones(shape, dtype=jnp.float64))

    @chex.variants(with_jit=True, without_jit=True)
    def test_large_finite_inputs_do_not_overflow(self):
        """The review's four-point example must sort identically at 1e200."""
        points = jnp.asarray(
            [[3.0, 1.0], [-2.0, 5.0], [0.0, -4.0], [1.0, 0.0]],
            dtype=jnp.float64,
        )
        base_order = self.variant(lambda: hilbert_sort(points))()

        scaled = points * 1e200
        scaled_order = self.variant(lambda: hilbert_sort(scaled))()

        chex.assert_trees_all_close(
            scaled_order, base_order, rtol=0.0, atol=0.0
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_very_small_finite_inputs_do_not_overflow(self):
        """Very small finite magnitudes must not collapse the ordering."""
        points = jnp.asarray(
            [[3.0, 1.0], [-2.0, 5.0], [0.0, -4.0], [1.0, 0.0]],
            dtype=jnp.float64,
        )
        base_order = self.variant(lambda: hilbert_sort(points))()

        scaled = points * 1e-200
        scaled_order = self.variant(lambda: hilbert_sort(scaled))()

        chex.assert_trees_all_close(
            scaled_order, base_order, rtol=0.0, atol=0.0
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_all_zero_column_maps_to_interval_centre(self):
        """A constant (all-zero) column must not produce NaN-dependent order."""
        points = jnp.asarray(
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
            dtype=jnp.float64,
        )
        order = self.variant(lambda: hilbert_sort(points))()
        # The constant column contributes no ordering information; the result
        # must be a valid permutation (no NaN collapse).
        chex.assert_trees_all_close(
            jnp.sort(order), jnp.arange(4), rtol=0.0, atol=0.0
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_mixed_magnitude_columns_are_handled(self):
        """Columns with very different magnitudes must not overflow."""
        points = jnp.asarray(
            [[1e200, 1.0], [-1e200, 2.0], [0.0, 3.0], [1e-200, 4.0]],
            dtype=jnp.float64,
        )
        order = self.variant(lambda: hilbert_sort(points))()
        chex.assert_trees_all_close(
            jnp.sort(order), jnp.arange(4), rtol=0.0, atol=0.0
        )
