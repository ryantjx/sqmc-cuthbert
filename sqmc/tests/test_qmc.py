"""Tests for the QMC point-set generators in ``sqmc.qmc.qmc``.

Test conventions follow ``state-space-models/cuthbert``: ``chex.TestCase``
classes, absl ``parameterized`` markers, module-level ``chex.assert_trees_all_close``
and ``chex.assert_shape``, and a module-autouse x64 fixture that restores the
flag on teardown.

``@chex.variants(with_jit, without_jit)`` is applied only to pure,
array-valued transforms (e.g. ``_apply_lms``). The engine ``sample()``
methods are stateful (they mutate ``_num_generated``) and are therefore run
as plain ``chex.TestCase`` methods.
"""

import sys
from pathlib import Path

import jax

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import jax.numpy as jnp
import numpy as np
import pytest
from absl.testing import parameterized
from scipy.stats import qmc

import chex

from sqmc.qmc.qmc import (
    Halton,
    Sobol,
    _MAXBITS,
    _apply_lms,
    normal_coordinates,
)


@pytest.fixture(scope="module", autouse=True)
def config():
    """Enable double precision for the module and restore it on teardown."""
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", False)


class HaltonTest(chex.TestCase):
    def test_known_first_points(self):
        actual = np.asarray(
            Halton(
                d=2,
                scramble=False,
                start_index=0,
                dtype=jnp.float64,
            ).sample(8)
        )
        expected = np.array(
            [
                [0.0, 0.0],
                [0.5, 1.0 / 3.0],
                [0.25, 2.0 / 3.0],
                [0.75, 1.0 / 9.0],
                [0.125, 4.0 / 9.0],
                [0.625, 7.0 / 9.0],
                [0.375, 2.0 / 9.0],
                [0.875, 5.0 / 9.0],
            ],
            dtype=np.float64,
        )

        chex.assert_trees_all_close(actual, expected, rtol=0.0, atol=1e-15)

    @parameterized.product(d=[1, 2, 5], n=[1, 100])
    def test_unscrambled_matches_scipy(self, d, n):
        actual = np.asarray(
            Halton(
                d=d,
                scramble=False,
                start_index=0,
                dtype=jnp.float64,
            ).sample(n)
        )
        expected = qmc.Halton(d=d, scramble=False).random(n)
        chex.assert_trees_all_close(actual, expected, rtol=0.0, atol=1e-15)

    @parameterized.product(scramble=[False, True])
    def test_chunked_sampling_matches_single_batch(self, scramble):
        whole = np.asarray(
            Halton(
                d=5,
                scramble=scramble,
                key=jax.random.PRNGKey(42),
                start_index=0,
                dtype=jnp.float64,
            ).sample(100)
        )

        engine = Halton(
            d=5,
            scramble=scramble,
            key=jax.random.PRNGKey(42),
            start_index=0,
            dtype=jnp.float64,
        )
        chunked = np.concatenate(
            [
                np.asarray(engine.sample(17)),
                np.asarray(engine.sample(33)),
                np.asarray(engine.sample(50)),
            ],
            axis=0,
        )

        chex.assert_trees_all_close(whole, chunked, rtol=0.0, atol=0.0)

    def test_scramble_key_controls_reproducibility(self):
        first = np.asarray(
            Halton(
                d=5,
                scramble=True,
                key=jax.random.PRNGKey(42),
                dtype=jnp.float64,
            ).sample(100)
        )
        repeated = np.asarray(
            Halton(
                d=5,
                scramble=True,
                key=jax.random.PRNGKey(42),
                dtype=jnp.float64,
            ).sample(100)
        )
        different = np.asarray(
            Halton(
                d=5,
                scramble=True,
                key=jax.random.PRNGKey(43),
                dtype=jnp.float64,
            ).sample(100)
        )

        chex.assert_trees_all_close(first, repeated, rtol=0.0, atol=0.0)
        self.assertFalse(np.array_equal(first, different))

    def test_owen_permutations_are_valid(self):
        engine = Halton(
            d=5,
            scramble=True,
            key=jax.random.PRNGKey(42),
            dtype=jnp.float64,
        )

        for base, num_digits, permutations in zip(
            engine._bases,
            engine._digits_per_dim,
            engine._permutations,
        ):
            permutations = np.asarray(permutations)
            chex.assert_shape(permutations, (num_digits, base))

            expected_digits = np.arange(base)
            for permutation in permutations:
                chex.assert_trees_all_close(
                    np.sort(permutation),
                    expected_digits,
                    rtol=0.0,
                    atol=0.0,
                )

    def test_scrambled_kernel_matches_scipy_with_same_permutations(self):
        scipy_engine = qmc.Halton(
            d=5,
            scramble=True,
            rng=np.random.default_rng(42),
        )
        custom_engine = Halton(
            d=5,
            scramble=True,
            key=jax.random.PRNGKey(42),
            start_index=0,
            dtype=jnp.float64,
        )

        # This private SciPy state is used only to isolate and validate the
        # custom radical-inverse kernel independently of RNG differences.
        custom_engine._permutations = tuple(
            jnp.asarray(permutation)
            for permutation in scipy_engine._permutations
        )

        actual = np.asarray(custom_engine.sample(100))
        expected = scipy_engine.random(100)
        chex.assert_trees_all_close(actual, expected, rtol=0.0, atol=0.0)

    @parameterized.parameters(0, -1, 10_001)
    def test_invalid_dimension_raises(self, d):
        with pytest.raises(ValueError):
            Halton(d=d)

    @parameterized.parameters(-1, -10)
    def test_negative_sample_size_raises(self, n):
        with pytest.raises(ValueError):
            Halton(d=2).sample(n)

    @parameterized.parameters(1.5, "10", None)
    def test_noninteger_sample_size_raises(self, n):
        with pytest.raises(TypeError):
            Halton(d=2).sample(n)

    @parameterized.product(scramble=[False, True])
    def test_explicit_state_scan_continues_sequence(self, scramble):
        """Three continuation calls of 8 points must match one 24-point call.

        The explicit-state path must advance the sequence index on every
        runtime iteration inside ``jax.lax.scan``, unlike the eager counter
        which only advances during tracing.
        """
        key = jax.random.PRNGKey(42)
        engine = Halton(
            d=5,
            scramble=scramble,
            key=key,
            start_index=0,
            dtype=jnp.float64,
        )

        def step(qmc_state, _):
            points, next_state = engine.sample(8, state=qmc_state)
            return next_state, points

        final_state, batches = jax.lax.scan(
            step, engine._eager_state(), None, length=3
        )
        chunked = batches.reshape(-1, 5)

        # Compiled-vs-compiled must match exactly. The eager path may differ
        # from the compiled path by XLA FMA fusion at the last bit, so compare
        # against a compiled single 24-point call instead.
        jit_whole = jax.jit(
            lambda s: Halton(
                d=5,
                scramble=scramble,
                key=key,
                start_index=0,
                dtype=jnp.float64,
            ).sample(24, state=s)[0]
        )(engine._eager_state())
        chex.assert_trees_all_close(chunked, jit_whole, rtol=0.0, atol=0.0)
        self.assertEqual(int(final_state.next_index), 24)

    @parameterized.product(scramble=[False, True])
    def test_explicit_state_is_reproducible(self, scramble):
        """Repeating a call with the same explicit state must reproduce the
        result without modifying the engine."""
        engine = Halton(
            d=5,
            scramble=scramble,
            key=jax.random.PRNGKey(7),
            start_index=0,
            dtype=jnp.float64,
        )
        state = engine._eager_state()
        first, next_state = engine.sample(8, state=state)
        repeated, repeated_state = engine.sample(8, state=state)

        chex.assert_trees_all_close(first, repeated, rtol=0.0, atol=0.0)
        self.assertEqual(int(next_state.next_index), int(repeated_state.next_index))
        # The engine's eager counter is untouched by the explicit-state path.
        self.assertEqual(engine._num_generated, 0)

    def test_explicit_state_does_not_advance_eager_counter(self):
        engine = Halton(
            d=5,
            scramble=False,
            start_index=0,
            dtype=jnp.float64,
        )
        state = engine._eager_state()
        engine.sample(8, state=state)
        self.assertEqual(engine._num_generated, 0)
        # The eager path still advances the counter.
        engine.sample(8)
        self.assertEqual(engine._num_generated, 8)

    @parameterized.parameters(-1, -10)
    def test_negative_start_index_raises(self, start_index):
        with pytest.raises(ValueError):
            Halton(d=2, start_index=start_index)

    @parameterized.parameters(1.5, "10", None, True)
    def test_noninteger_start_index_raises(self, start_index):
        with pytest.raises(TypeError):
            Halton(d=2, start_index=start_index)

    def test_start_index_2_to_32_raises(self):
        # 2**32 wraps to 0 in uint32, which would silently reproduce the
        # beginning of the sequence. It must be rejected up front.
        with pytest.raises(ValueError):
            Halton(d=2, start_index=2**32)

    def test_block_crossing_index_limit_raises(self):
        # d=1 uses base 2. With float64, digits_per_dim gives a limit of
        # 2**53, so a block ending exactly at the limit is allowed but the
        # next point is rejected. Use tiny batches near the limit rather than
        # allocating the whole sequence.
        engine = Halton(d=1, scramble=False, start_index=0, dtype=jnp.float64)
        limit = engine._index_limit()
        # A block ending exactly at the limit is permitted.
        engine._num_generated = limit - 2
        engine.sample(2)
        # The next point must be rejected.
        with pytest.raises(ValueError):
            engine.sample(1)

    def test_last_valid_indices_match_radical_inverse(self):
        # Exercise the last valid indices in tiny batches and check them
        # against an independent radical-inverse calculation, without
        # allocating the whole sequence.
        engine = Halton(d=1, scramble=False, start_index=0, dtype=jnp.float64)
        limit = engine._index_limit()
        start = limit - 4
        engine._num_generated = start
        points = np.asarray(engine.sample(4))

        # Independent radical inverse of the last four indices in base 2.
        expected = []
        for idx in range(start, limit):
            value = 0.0
            factor = 0.5
            while idx:
                value += (idx % 2) * factor
                idx //= 2
                factor *= 0.5
            expected.append(value)
        chex.assert_trees_all_close(
            points[:, 0], np.asarray(expected), rtol=0.0, atol=1e-15
        )


class SobolTest(chex.TestCase):
    def test_known_first_points(self):
        # ``Sobol.sample`` retains the origin by default (``start_index=0``),
        # so the first returned point is the origin.
        actual = np.asarray(
            Sobol(d=2, scramble=False, dtype=jnp.float64).sample(8)
        )
        expected = np.array(
            [
                [0.0, 0.0],
                [0.5, 0.5],
                [0.75, 0.25],
                [0.25, 0.75],
                [0.375, 0.375],
                [0.875, 0.875],
                [0.625, 0.125],
                [0.125, 0.625],
            ],
            dtype=np.float64,
        )

        chex.assert_trees_all_close(actual, expected, rtol=0.0, atol=0.0)

    @parameterized.product(d=[1, 2, 5], m=[0, 3, 7])
    def test_unscrambled_matches_scipy(self, d, m):
        n = 2**m
        actual = np.asarray(
            Sobol(d=d, scramble=False, dtype=jnp.float64).sample(n)
        )
        # scipy also retains the origin, so the sequences agree directly.
        scipy_points = qmc.Sobol(d=d, scramble=False, bits=_MAXBITS).random(n)
        expected = scipy_points
        chex.assert_trees_all_close(actual, expected, rtol=0.0, atol=0.0)

    @parameterized.product(scramble=[False, True])
    def test_chunked_sampling_matches_single_batch(self, scramble):
        key = jax.random.PRNGKey(42)
        whole = np.asarray(
            Sobol(
                d=5,
                scramble=scramble,
                key=key,
                dtype=jnp.float64,
            ).sample(64)
        )

        engine = Sobol(
            d=5,
            scramble=scramble,
            key=key,
            dtype=jnp.float64,
        )
        chunked = np.concatenate(
            [
                np.asarray(engine.sample(7)),
                np.asarray(engine.sample(19)),
                np.asarray(engine.sample(38)),
            ],
            axis=0,
        )

        chex.assert_trees_all_close(whole, chunked, rtol=0.0, atol=0.0)

    def test_scramble_key_controls_reproducibility(self):
        first = np.asarray(
            Sobol(
                d=5,
                scramble=True,
                key=jax.random.PRNGKey(42),
                dtype=jnp.float64,
            ).sample(64)
        )
        repeated = np.asarray(
            Sobol(
                d=5,
                scramble=True,
                key=jax.random.PRNGKey(42),
                dtype=jnp.float64,
            ).sample(64)
        )
        different = np.asarray(
            Sobol(
                d=5,
                scramble=True,
                key=jax.random.PRNGKey(43),
                dtype=jnp.float64,
            ).sample(64)
        )

        chex.assert_trees_all_close(first, repeated, rtol=0.0, atol=0.0)
        self.assertFalse(np.array_equal(first, different))

    def test_lms_row_masks_are_unit_lower_triangular(self):
        engine = Sobol(
            d=5,
            scramble=True,
            key=jax.random.PRNGKey(42),
        )
        row_masks = np.asarray(engine._lms_matrices, dtype=np.uint32)

        rows = np.arange(_MAXBITS, dtype=np.uint32)
        bit_positions = np.uint32(_MAXBITS - 1) - rows
        diagonal_masks = np.left_shift(np.uint32(1), bit_positions)
        allowed_masks = np.left_shift(
            np.left_shift(np.uint32(1), rows + np.uint32(1)) - np.uint32(1),
            bit_positions,
        )

        chex.assert_shape(row_masks, (engine.d, _MAXBITS))
        chex.assert_trees_all_close(
            row_masks & diagonal_masks[None, :],
            np.broadcast_to(diagonal_masks, row_masks.shape),
            rtol=0.0,
            atol=0.0,
        )
        chex.assert_trees_all_close(
            row_masks & ~allowed_masks[None, :],
            np.zeros_like(row_masks),
            rtol=0.0,
            atol=0.0,
        )

    @chex.variants(with_jit=True, without_jit=True)
    def test_identity_lms_preserves_direction_integers(self):
        engine = Sobol(d=5, scramble=False)
        bit_positions = jnp.arange(
            _MAXBITS - 1,
            -1,
            -1,
            dtype=jnp.uint32,
        )
        identity_row_masks = jnp.broadcast_to(
            (jnp.uint32(1) << bit_positions)[None, :],
            (engine.d, _MAXBITS),
        )

        actual = self.variant(_apply_lms)(
            engine._direction_integers,
            identity_row_masks,
        )
        chex.assert_trees_all_close(
            actual,
            engine._direction_integers,
            rtol=0.0,
            atol=0.0,
        )

    def test_first_scrambled_point_is_digital_shift(self):
        engine = Sobol(
            d=5,
            scramble=True,
            key=jax.random.PRNGKey(42),
            dtype=jnp.float64,
        )

        # ``Sobol.sample`` retains the origin, so the first returned point is
        # index 0: the digital shift alone (no direction integer).
        actual = np.asarray(engine.sample(1)[0])
        expected = (
            np.asarray(engine._digital_shift, dtype=np.float64)
            * 2.0 ** -_MAXBITS
        )
        chex.assert_trees_all_close(actual, expected, rtol=0.0, atol=0.0)

    def test_reset_reproduces_points(self):
        engine = Sobol(
            d=5,
            scramble=True,
            key=jax.random.PRNGKey(42),
            dtype=jnp.float64,
        )
        first = np.asarray(engine.sample(32))
        returned = engine.reset()
        repeated = np.asarray(engine.sample(32))

        self.assertIs(returned, engine)
        chex.assert_trees_all_close(first, repeated, rtol=0.0, atol=0.0)

    @parameterized.parameters(0, -1, 21_202)
    def test_invalid_dimension_raises(self, d):
        with pytest.raises(ValueError):
            Sobol(d=d)

    @parameterized.parameters(0, -1, -10)
    def test_nonpositive_sample_size_raises(self, n):
        with pytest.raises(ValueError):
            Sobol(d=2).sample(n)

    @parameterized.parameters(1.5, "10", None)
    def test_noninteger_sample_size_raises(self, n):
        with pytest.raises(TypeError):
            Sobol(d=2).sample(n)

    @parameterized.product(scramble=[False, True])
    def test_explicit_state_scan_continues_sequence(self, scramble):
        """Three continuation calls of 8 points must match one 24-point call.

        The explicit-state path must advance the sequence index on every
        runtime iteration inside ``jax.lax.scan``, unlike the eager counter
        which only advances during tracing.
        """
        key = jax.random.PRNGKey(42)
        engine = Sobol(
            d=5,
            scramble=scramble,
            key=key,
            dtype=jnp.float64,
        )

        def step(qmc_state, _):
            points, next_state = engine.sample(8, state=qmc_state)
            return next_state, points

        final_state, batches = jax.lax.scan(
            step, engine._eager_state(), None, length=3
        )
        chunked = batches.reshape(-1, 5)

        # All Sobol paths go through the same compiled kernel, so the eager
        # single call and the scan continuation must match exactly.
        whole = np.asarray(
            Sobol(
                d=5,
                scramble=scramble,
                key=key,
                dtype=jnp.float64,
            ).sample(24)
        )
        chex.assert_trees_all_close(chunked, whole, rtol=0.0, atol=0.0)
        # Sobol retains the origin, so 24 points end at exclusive index 24.
        self.assertEqual(int(final_state.next_index), 24)

    @parameterized.product(scramble=[False, True])
    def test_explicit_state_is_reproducible(self, scramble):
        """Repeating a call with the same explicit state must reproduce the
        result without modifying the engine."""
        engine = Sobol(
            d=5,
            scramble=scramble,
            key=jax.random.PRNGKey(7),
            dtype=jnp.float64,
        )
        state = engine._eager_state()
        first, next_state = engine.sample(8, state=state)
        repeated, repeated_state = engine.sample(8, state=state)

        chex.assert_trees_all_close(first, repeated, rtol=0.0, atol=0.0)
        self.assertEqual(int(next_state.next_index), int(repeated_state.next_index))
        # The engine's eager counter is untouched by the explicit-state path.
        self.assertEqual(engine._num_generated, 0)

    def test_explicit_state_does_not_advance_eager_counter(self):
        engine = Sobol(d=5, scramble=False, dtype=jnp.float64)
        state = engine._eager_state()
        engine.sample(8, state=state)
        self.assertEqual(engine._num_generated, 0)
        # The eager path still advances the counter.
        engine.sample(8)
        self.assertEqual(engine._num_generated, 8)

    @parameterized.product(start_index=[0, 1, 4])
    def test_start_index_controls_first_point(self, start_index):
        """The first generated index must be ``start_index`` (not always 1)."""
        engine = Sobol(
            d=1,
            scramble=False,
            dtype=jnp.float64,
            start_index=start_index,
        )
        first = np.asarray(engine.sample(1)[0])
        # Unscrambled Sobol' point at index i is the radical inverse of the
        # Gray code of i. Compare against scipy's sequence at that index.
        scipy_points = qmc.Sobol(d=1, scramble=False, bits=_MAXBITS).random(
            start_index + 1
        )
        expected = scipy_points[start_index]
        chex.assert_trees_all_close(first, expected, rtol=0.0, atol=0.0)

    def test_start_index_zero_includes_origin(self):
        """With ``start_index=0`` the first point is the origin (0, ..., 0)."""
        engine = Sobol(
            d=2,
            scramble=False,
            dtype=jnp.float64,
            start_index=0,
        )
        first = np.asarray(engine.sample(1)[0])
        chex.assert_trees_all_close(
            first, np.zeros(2), rtol=0.0, atol=0.0
        )

    def test_start_index_zero_balanced_strata(self):
        """Indices 0..3 give one point per first-coordinate quarter."""
        engine = Sobol(
            d=1,
            scramble=False,
            dtype=jnp.float64,
            start_index=0,
        )
        points = np.asarray(engine.sample(4))
        # First coordinate strata: [0,1/4), [1/4,1/2), [1/2,3/4), [3/4,1).
        strata = np.floor(points[:, 0] * 4).astype(int)
        chex.assert_trees_all_close(
            np.sort(strata), np.arange(4), rtol=0.0, atol=0.0
        )

    def test_start_index_one_breaks_balance(self):
        """Indices 1..4 leave the first quarter empty (the review's example)."""
        engine = Sobol(
            d=1,
            scramble=False,
            dtype=jnp.float64,
            start_index=1,
        )
        points = np.asarray(engine.sample(4))
        strata = np.floor(points[:, 0] * 4).astype(int)
        # The first quarter is empty and the second has two points.
        self.assertNotIn(0, strata.tolist())
        self.assertEqual(int(np.sum(strata == 1)), 2)

    def test_balanced_block_validation_passes_for_aligned_power_of_two(self):
        engine = Sobol(
            d=2,
            scramble=False,
            dtype=jnp.float64,
            start_index=0,
        )
        # start_index=0, n=8: aligned power-of-two block is valid.
        engine._validate_balanced_block(8)

    def test_balanced_block_validation_rejects_non_power_of_two(self):
        engine = Sobol(
            d=2,
            scramble=False,
            dtype=jnp.float64,
            start_index=0,
        )
        with pytest.raises(ValueError):
            engine._validate_balanced_block(6)

    def test_balanced_block_validation_rejects_unaligned_start(self):
        engine = Sobol(
            d=2,
            scramble=False,
            dtype=jnp.float64,
            start_index=1,
        )
        # start_index=1 is not divisible by n=8, so the block is unaligned.
        with pytest.raises(ValueError):
            engine._validate_balanced_block(8)

    @parameterized.parameters(-1, -10)
    def test_negative_start_index_raises(self, start_index):
        with pytest.raises(ValueError):
            Sobol(d=2, start_index=start_index)

    @parameterized.parameters(1.5, "10", None, True)
    def test_noninteger_start_index_raises(self, start_index):
        with pytest.raises(TypeError):
            Sobol(d=2, start_index=start_index)

    @parameterized.product(dtype=[jnp.float32, jnp.float64])
    def test_float32_high_index_stays_below_one(self, dtype):
        """At the reported high index, float32 must not round to 1.0."""
        # Index 715827882 is a valid 30-bit Sobol' index whose first
        # coordinate is 0.9999999990686774 in float64 and rounds to 1.0 in
        # float32. Set the counter directly without allocating the prefix.
        engine = Sobol(d=1, scramble=False, dtype=dtype, start_index=0)
        engine._num_generated = 715827882
        points = np.asarray(engine.sample(1))
        self.assertLess(points[0, 0], 1.0)
        self.assertGreaterEqual(points[0, 0], 0.0)

    @parameterized.product(dtype=[jnp.float32, jnp.float64])
    def test_generated_values_stay_below_one(self, dtype):
        """All generated Sobol' coordinates must satisfy the [0, 1) contract."""
        engine = Sobol(d=5, scramble=False, dtype=dtype, start_index=0)
        points = np.asarray(engine.sample(1024))
        self.assertLess(float(points.max()), 1.0)
        self.assertGreaterEqual(float(points.min()), 0.0)

    @parameterized.product(dtype=[jnp.float32, jnp.float64])
    def test_interior_values_unchanged_by_endpoint_policy(self, dtype):
        """The clipping must not alter interior (non-endpoint) values."""
        engine = Sobol(d=5, scramble=False, dtype=dtype, start_index=0)
        points = np.asarray(engine.sample(1024))
        # No interior value should equal the clipped upper bound unless it was
        # genuinely rounded to one (which the policy caps). All values strictly
        # below nextafter(1,0) are untouched.
        one = np.asarray(1, dtype=np.float32 if dtype == jnp.float32 else np.float64)
        upper = np.nextafter(one, np.asarray(0, dtype=one.dtype))
        interior = points[points < upper]
        # Interior values are all < 1 and >= 0, and none equals the cap.
        self.assertTrue(np.all(interior < 1.0))
        self.assertTrue(np.all(interior >= 0.0))

    @parameterized.product(dtype=[jnp.float32, jnp.float64])
    def test_normal_coordinates_finite_at_boundaries(self, dtype):
        """Exact zero and rounded-one inputs must give finite normal quantiles."""
        zero = jnp.asarray(0.0, dtype=dtype)
        one = jnp.asarray(1.0, dtype=dtype)
        upper = jnp.nextafter(one, zero)
        for u in (zero, upper):
            q = normal_coordinates(u)
            self.assertTrue(bool(jnp.isfinite(q)))

    @parameterized.product(dtype=[jnp.float32, jnp.float64])
    def test_normal_coordinates_interior_unchanged(self, dtype):
        """Interior uniforms map to the standard normal quantile unchanged."""
        u = jnp.asarray([0.1, 0.5, 0.9], dtype=dtype)
        q = normal_coordinates(u)
        expected = jax.scipy.special.ndtri(u)
        chex.assert_trees_all_close(q, expected, rtol=1e-6, atol=1e-6)
