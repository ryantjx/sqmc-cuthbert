"""
Module for Quasi-Monte Carlo (QMC) sampling methods. This module provides the interface and classes for QMC methods:

- QMC
- Halton(QMC)
- Sobol(QMC)

Introduction to Quasi-Monte Carlo (QMC) methods:
================================================

Quasi-Monte Carlo (QMC) methods [1]_,[2]_, [3]_ provide an array of :math:`n \times d` points in :math:`[0,1]^d`. Compared to random points, QMC points are designed to be more evenly distributed with fewer gaps and clusters. This is quantified by discrepancy measures [4]_. From the Koksma-Hlawka inequality [5]_ we know that low discrepancy reduces a bound on integration error. Averaging a function :math:`f` over :math:`n` QMC points can achieve an integration error close to :math:`O(n^{-1})` for well behaved functions [2]_.

References
----------
.. [1] Owen, Art B. "Monte Carlo Book: the Quasi-Monte Carlo parts." 2019.
.. [2] Niederreiter, Harald. "Random number generation and quasi-Monte Carlo
   methods." Society for Industrial and Applied Mathematics, 1992.
.. [3] Dick, Josef, Frances Y. Kuo, and Ian H. Sloan. "High-dimensional
   integration: the quasi-Monte Carlo way." Acta Numerica no. 22: 133, 2013.
.. [4] Aho, A. V., C. Aistleitner, T. Anderson, K. Appel, V. Arnol'd, N.
   Aronszajn, D. Asotsky et al. "W. Chen et al.(eds.), "A Panorama of
   Discrepancy Theory", Sringer International Publishing,
   Switzerland: 679, 2014.
.. [5] Hickernell, Fred J. "Koksma-Hlawka Inequality." Wiley StatsRef:
   Statistics Reference Online, 2014.


Note: Implementation is based on scipy.stats.qmc (https://docs.scipy.org/doc/scipy/reference/stats.qmc.html) and QuasiMonteCarlo.jl (https://github.com/SciML/QuasiMonteCarlo.jl).
"""


from abc import abstractmethod
from functools import partial
from typing import ClassVar, NamedTuple
import math

import jax
import jax.numpy as jnp
import numpy as np
import os

jax.config.update("jax_enable_x64", True)
# jax.config.update('jax_platform_name', 'cpu')

_MAX_DIMENSION_HALTON = 10000


class QMCState(NamedTuple):
    """Explicit sampling state for a QMC engine.

    ``next_index`` is the next sequence index to generate (the exclusive end of
    the previously generated block). It is stored as a ``uint32`` array so the
    state is a valid JAX pytree that can be carried through ``jax.lax.scan``.
    """

    next_index: jax.Array


class QMC:
    """
    Interface for QMC classs.

    Args:
    """
    # Declared on the base so subclasses and consumers can rely on them.
    scramble: bool
    dtype: jnp.dtype

    def __init__(self, d: int) -> None:
        self._initialize(d=d)
    
    def _initialize(self, d: int) -> None:
        self.d = d
    
    @abstractmethod
    def sample(
        self, n: int, *, state: QMCState | None = None
    ) -> jnp.ndarray | tuple[jnp.ndarray, QMCState]:
        raise NotImplementedError("sample method must be implemented in subclasses.")


class Halton(QMC):
    """The Halton sequence uses a radical-inverse sequence in a distinct prime base for each dimension [1]_. When scrambling is enabled, an independent random permutation is applied to the digits at each digit position and in each dimension, following the randomized Halton algorithm of Owen [2]_.

    References
    ----------
    .. [1] J. H. Halton, "On the efficiency of certain quasi-random
    sequences of points in evaluating multi-dimensional integrals,"
    Numerische Mathematik, 2, 84-90, 1960.
    doi:10.1007/BF01386213

    .. [2] A. B. Owen, "A randomized Halton algorithm in R,"
    arXiv:1706.02808, 2017.
    doi:10.48550/arXiv.1706.02808
    """
    def __init__(
        self, 
        d: int, 
        scramble: bool = False,
        key: jax.Array = jax.random.PRNGKey(0),
        start_index: int = 0,
        dtype: jnp.dtype = jnp.float64
    ):
        """_summary_

        Args:
            d (int): number of dimensions.
            scramble (bool, optional): Whether to scramble Halton sequence. If True, applies . Defaults to False.
        """
        if not 1 <= d <= _MAX_DIMENSION_HALTON:
            raise ValueError(f"Dimension d must be in [1, {_MAX_DIMENSION_HALTON}]")

        if not isinstance(d, int):
            raise TypeError("d must be a Python integer.")

        if isinstance(start_index, bool) or not isinstance(start_index, int):
            raise TypeError("start_index must be a Python integer.")
        if start_index < 0:
            raise ValueError("start_index must be nonnegative.")

        super().__init__(d=d)
        self.scramble = bool(scramble)
        self._bases = tuple(int(p) for p in _PRIMES[:d])
        self.key = key
        self.start_index = start_index
        self._num_generated = 0
        self.dtype = dtype

        if self.scramble:
            (
                self._permutations,
                self._tail_corrections,
                self._digits_per_dim,
            ) = self._initialize_owen_scramble()
        else:
            self._permutations = None
            self._tail_corrections = None
            self._digits_per_dim = self._initialize_digits_per_dim()

        # A start at or beyond the index limit would wrap in uint32 and
        # silently reproduce the beginning of the sequence. Reject it up front.
        if start_index >= self._index_limit():
            raise ValueError(
                f"start_index must be below the index limit "
                f"{self._index_limit()}, but got {start_index}."
            )

    def _initialize_digits_per_dim(self):
        precision_bits = 23 if self.dtype == jnp.float32 else 53
        # precision_bits + 1 is the threshold bits
        return tuple(max(1, math.ceil((precision_bits + 1) / math.log2(base)) - 1) for base in self._bases)
    
    def _initialize_owen_scramble(self):
        digits_per_dim = self._initialize_digits_per_dim()
        all_permutations = []
        tail_corrections = []
        for dim, (base, num_digits) in enumerate(zip(self._bases, digits_per_dim)):

            dim_key = jax.random.fold_in(self.key, dim)
            digit_keys = jax.vmap(lambda k: jax.random.fold_in(dim_key, k))(jnp.arange(num_digits, dtype=jnp.uint32))
            permutations = jax.vmap(lambda k: jax.random.permutation(k, base))(digit_keys)
            tail_key = jax.random.fold_in(dim_key, num_digits)
            tail = (jax.random.uniform(tail_key, shape=(), minval=0.0, maxval=1.0, dtype=self.dtype) * base ** (-num_digits))

            all_permutations.append(permutations)
            tail_corrections.append(tail)
        return (tuple(all_permutations), jnp.stack(tail_corrections), tuple(digits_per_dim))

    def _radical_inverse(
        self,
        indices: jnp.ndarray,
        base: int,
        num_digits: int,
    ):
        """Compute a vectorized, deterministic radical inverse."""
        base_integers = jnp.asarray(base, dtype=indices.dtype)
        base_floats = jnp.asarray(base, dtype=self.dtype)

        initial_state = (indices, jnp.zeros_like(indices, dtype=self.dtype), jnp.asarray(1.0, dtype=self.dtype) / base_floats)

        def step(state, _):
            remaining, value, factor = state
            digit = remaining % base_integers
            values = value + digit.astype(self.dtype) * factor
            remaining = remaining // base_integers
            factor = factor / base_floats
            return (remaining, values, factor), None
        final_state, _ = jax.lax.scan(f=step, init=initial_state, xs=None, length=num_digits)

        _, values, _ = final_state
        return values
    
    def _scrambled_radical_inverse(
        self,
        indices: jnp.ndarray,
        base: int,
        permutations: jnp.ndarray,
        tail_correction: float,
    ):
        """Compute a vectorized, scrambled radical inverse."""
        base_integers = jnp.asarray(base, dtype=indices.dtype)
        base_floats = jnp.asarray(base, dtype=self.dtype)

        initial_state = (indices, jnp.zeros_like(indices, dtype=self.dtype), jnp.asarray(1.0, dtype=self.dtype) / base_floats)

        def step(state, permutation):
            remaining, value, factor = state
            digit = remaining % base_integers
            # apply position permutation to every point digit.
            scrambled_digit = permutation[digit]
            values = (value + scrambled_digit.astype(self.dtype) * factor)
            remaining = remaining // base_integers
            factor = factor / base_floats
            return (remaining, values, factor), None
        final_state, _ = jax.lax.scan(f=step, init=initial_state, xs=permutations, length=permutations.shape[0])
        _, values, _ = final_state
        return values
        # return values + tail_correction
    def _index_limit(self) -> int:
        """Maximum exclusive index representable without wraparound.

        The minimum of ``2**32`` (the ``uint32`` index dtype) and the supported
        digit range of every active coordinate. A final block may end exactly
        at this limit, but the next point is rejected.
        """
        return min(
            2**32,
            *(base**digits for base, digits in zip(self._bases, self._digits_per_dim)),
        )

    def _eager_state(self) -> QMCState:
        """Return the current sampling state for the eager path."""
        return QMCState(
            next_index=jnp.asarray(
                self.start_index + self._num_generated, dtype=jnp.uint32
            )
        )

    def _sample_from_state(self, n: int, state: QMCState):
        """Generate ``n`` points continuing from ``state``.

        Pure computation shared by the eager and explicit-state paths. It does
        not read or update ``_num_generated`` or any other captured Python
        attribute, so it is safe to trace inside ``jax.lax.scan``.

        Returns ``(points, next_state)``.
        """
        first_index = state.next_index
        # Offset form keeps the batch size static while allowing a dynamic
        # (traced) start index inside a compiled scan.
        indices = jnp.arange(n, dtype=jnp.uint32) + first_index

        coordinates = []

        for dim, (base, num_digits) in enumerate(zip(self._bases, self._digits_per_dim)):
            if self.scramble:
                coordinate = self._scrambled_radical_inverse(
                    indices=indices,
                    base=base,
                    permutations=self._permutations[dim],
                    tail_correction=(self._tail_corrections[dim])
                )
            else:
                coordinate = self._radical_inverse(
                    indices=indices,
                    base=base,
                    num_digits=num_digits
                )
            coordinates.append(coordinate)

        points = jnp.stack(coordinates, axis=-1).astype(self.dtype)
        return points, QMCState(next_index=first_index + n)

    def sample(
        self, n: int, *, state: QMCState | None = None
    ) -> jnp.ndarray | tuple[jnp.ndarray, QMCState]:
        """Halton sequence generator.

        Args:
            n (int): Number of samples to generate.
            state (QMCState | None): Optional explicit sampling state. When
                provided, returns ``(points, next_state)`` and does not mutate
                the engine's internal counter, so it can be used inside a
                compiled ``jax.lax.scan``. When ``None`` (default), returns
                only ``points`` and advances the engine's counter.

        Returns:
            jnp.ndarray: An array of shape (n, d) containing the Halton
            sequence samples, or ``(points, next_state)`` when ``state`` is
            given.
        """
        if not isinstance(n, int):
            raise TypeError("n must be a Python integer.")

        if n <= 0:
            raise ValueError("Number of samples n must be positive.")

        if state is not None:
            return self._sample_from_state(n, state)

        # Eager path: validate bounds before generating. Use wrap-safe
        # arithmetic so an overflowing start cannot silently wrap to a small
        # index and reproduce the beginning of the sequence.
        first_index = self.start_index + self._num_generated
        stop_index = first_index + n

        limit = self._index_limit()
        if first_index < 0 or n > limit or first_index > limit - n:
            raise ValueError(
                f"Requested Halton block exceeds the index range. "
                f"Maximum exclusive index is {limit}, but requested "
                f"[{first_index}, {stop_index}). Consider reducing n or "
                f"increasing precision."
            )

        points, _ = self._sample_from_state(n, self._eager_state())
        self._num_generated += n
        return points

def _primes_less_than(n: int) -> np.ndarray:
    """Generate first n prime numbers in jax array. Generated with numpy Sieve of Eratosthenes."""
    sieve = np.ones(n // 2, dtype=bool)
    for i in range(3, int(n**0.5) + 1, 2):
        if sieve[i // 2]:
            sieve[i*i//2::i] = False
    primes = 2 * np.where(sieve)[0] + 1
    primes[0] = 2
    return primes[primes > 0]

# 10,000th prime is 104729, so we generate primes up to 104729 + 1
_PRIMES = _primes_less_than(104729 + 1)
# check that we have exactly 10,000 primes
assert len(_PRIMES) == _MAX_DIMENSION_HALTON

######################################

_MAXBITS = 30  # Direction integers file has 30 columns
SCALE = 2**(-_MAXBITS)
MAX_POINTS = 2**(_MAXBITS)
_DIRECTION_INTEGERS = np.load(os.path.join(os.path.dirname(__file__), '_sobol_direction_numbers.npz'))['direction_integers']
class Sobol(QMC):
    """
    Sobol' sequence generator using the construction introduced by Sobol' [1]_ and the Joe--Kuo direction numbers [4]_. Randomization is performed using a left linear matrix scramble followed by a digital random shift (LMS+shift) [5]_. This is the scrambling strategy used by ``scipy.stats.qmc.Sobol``. For the more general nested scrambling framework, see Owen [6]_.

    Args:
        d (int): Dimension of the Sobol sequence.
        scramble (bool): Whether to scramble the Sobol sequence. If True, a random scrambling is applied to the sequence.

    References
    ---------
    .. [1] I. M. Sobol', "On the distribution of points in a cube and the approximate evaluation of integrals," USSR Computational Mathematics and Mathematical Physics, 7(4), 86-112, 1967. doi:10.1016/0041-5553(67)90144-9

    .. [4] S. Joe and F. Y. Kuo, "Constructing Sobol' sequences with better
    two-dimensional projections," SIAM Journal on Scientific Computing, 30(5), 2635-2654, 2008. doi:10.1137/070709359

    .. [5] J. Matousek, "On the L2-discrepancy for anchored boxes,"
    Journal of Complexity, 14(4), 527-556, 1998.
    doi:10.1006/jcom.1998.0489

    .. [6] A. B. Owen, "Randomly permuted (t,m,s)-nets and (t,s)-sequences," in Monte Carlo and Quasi-Monte Carlo Methods in Scientific Computing, Lecture Notes in Statistics, vol. 106, pp. 299-317, Springer, 1995. doi:10.1007/978-1-4612-2552-2_19
    """
    def __init__(self, 
        d: int, 
        scramble: bool = False,
        key: jax.Array = jax.random.PRNGKey(0),
        dtype: jnp.dtype = jnp.float64,
        start_index: int = 0
    ):
        max_dimension = _DIRECTION_INTEGERS.shape[0]

        if not isinstance(d, int):
            raise TypeError("d must be a Python integer.")

        if not 1 <= d <= max_dimension:
            raise ValueError(
                f"Dimension d must be in [1, {max_dimension}]."
            )

        if dtype not in (jnp.float32, jnp.float64):
            raise ValueError("dtype must be jnp.float32 or jnp.float64.")

        if isinstance(start_index, bool) or not isinstance(start_index, int):
            raise TypeError("start_index must be a Python integer.")
        if start_index < 0:
            raise ValueError("start_index must be nonnegative.")

        super().__init__(d=d)
        self.scramble = bool(scramble)
        self.key=key
        self.dtype = dtype
        self.start_index = start_index
        self._num_generated = 0

        self._direction_integers = jnp.asarray(_DIRECTION_INTEGERS[0:d, 0:_MAXBITS], dtype=jnp.uint32)
        
        if self.scramble:
            (self._direction_integers, self._digital_shift, self._lms_matrices) = self._initialize_scramble()
        else:
            self._digital_shift = jnp.zeros(shape=(self.d,), dtype=jnp.uint32)
            self._lms_matrices = None

    def _initialize_scramble(self):
        self.key, lms_key, shift_key = jax.random.split(self.key, 3)

        # lms_matrices = self._initialize_lms_matrices(lms_key)
        lms_row_masks = self._initialize_lms_row_masks(lms_key)

        scrambled_direction_integers = _apply_lms(
            direction_integers=self._direction_integers,
            lms_row_masks=lms_row_masks
        )
        digital_shift = self._initialize_digital_shift(shift_key)
        return (scrambled_direction_integers, digital_shift, lms_row_masks)

    def _initialize_lms_row_masks(self, key: jax.Array) -> jnp.ndarray:
        """Generate packed unit lower-triangular row masks."""
        rows = jnp.arange(_MAXBITS, dtype=jnp.uint32)
        bit_positions = jnp.uint32(_MAXBITS -1) - rows
        diagonal_mask = jnp.uint32(1) << bit_positions
        allowed_mask = ((jnp.uint32(1) << (rows + jnp.uint32(1))) - jnp.uint32(1))  << bit_positions
        random_masks = jax.random.bits(key, shape=(self.d, _MAXBITS), dtype=jnp.uint32)
        # remove bits above the LMS diagonal and set diagonal to 1. This ensures that the LMS matrix is unit lower-triangular and invertible over GF(2).
        row_masks = (random_masks & allowed_mask[None, :]) | diagonal_mask[None, :] 
        return row_masks
    
    # NOTE: this implementation is not vectorized and is slow.
    # def _initialize_lms_matrices(self, key: jax.Array) -> jax.Array:
    #     """
    #     Generate one random unit lower-triangular binary matrix per dimension. The diagonal must be one so that every matrix is invertible over GF(2).
    #     """
    #     matrices = jax.random.randint(
    #         key,
    #         shape=(self.d, _MAXBITS, _MAXBITS),
    #         minval=0,
    #         maxval=2,
    #         dtype=jnp.uint32,
    #     )

    #     # Remove everything above the diagonal.
    #     matrices = jnp.tril(matrices)
    #     diagonal = jnp.arange(_MAXBITS)
    #     # Force the diagonal to one.
    #     matrices = matrices.at[:, diagonal, diagonal].set(
    #         jnp.uint32(1)
    #     )
    #     return matrices
    # def _apply_lms(self, direction_integers: jnp.ndarray, lms_matrices: jnp.ndarray) -> jnp.ndarray:
    #     """Apply each dimension's LMS matrix to corresponding direction integers. All operations are performed in GF(2) (i.e., modulo 2)."""
    #     # 
    #     shifts = jnp.arange(_MAXBITS -1, -1, -1, dtype=jnp.uint32) 

    #     # Shape: (d, direction_number, input_bit)
    #     direction_bits = (direction_integers[:, :, None] >> shifts[None, None, :]) & jnp.uint32(1)
    #     # Matrix-vector multiplication over GF(2):
        
    #     # output_bit[r] = XOR_s lms_matrix[r, s] * input_bit[s]
    #     # Shape: (d, direction_number, output_bit)
    #     scrambled_bits = jnp.sum((lms_matrices[:, None, :, :] * direction_bits[:, :, None, :]), axis=-1, dtype=jnp.uint32 ) & jnp.uint32(1)

    #     # Pack the binary vectors back into uint32 direction integers.
    #     bit_weights = (jnp.uint32(1) << shifts)

    #     scrambled_direction_integers = jnp.sum(scrambled_bits * bit_weights[None, None, :],axis=-1,dtype=jnp.uint32)

    #     return scrambled_direction_integers
        
    def _initialize_digital_shift(self, key: jax.Array) -> jnp.ndarray:
        """
        Generate one uniformly random _MAXBITS-bit digital shift
        per dimension.
        """
        shift_bits = jax.random.randint(key, shape=(self.d, _MAXBITS), minval=0, maxval=2, dtype=jnp.uint32)

        shifts = jnp.arange(_MAXBITS - 1, -1, -1, dtype=jnp.uint32)

        bit_weights = jnp.uint32(1) << shifts

        return jnp.sum(shift_bits * bit_weights[None, :], axis=-1, dtype=jnp.uint32)
    
    def _eager_state(self) -> QMCState:
        """Return the current sampling state for the eager path.

        The first generated index is ``start_index + _num_generated``. The
        default ``start_index=0`` retains the origin, matching the balanced-net
        convention used throughout the SQMC experiments.
        """
        return QMCState(
            next_index=jnp.asarray(
                self.start_index + self._num_generated, dtype=jnp.uint32
            )
        )

    def _sample_from_state(self, n: int, state: QMCState):
        """Generate ``n`` points continuing from ``state``.

        Pure computation shared by the eager and explicit-state paths. It does
        not read or update ``_num_generated`` or any other captured Python
        attribute, so it is safe to trace inside ``jax.lax.scan``.

        Returns ``(points, next_state)``.
        """
        first_index = state.next_index
        points = _sobol_sample_batched(
            first_index=first_index,
            n=n,
            direction_integers=self._direction_integers,
            digital_shift=self._digital_shift,
            num_bits=_MAXBITS,
            dtype=self.dtype,
        )
        return points, QMCState(next_index=first_index + n)

    def sample(
        self, n: int, *, state: QMCState | None = None
    ) -> jnp.ndarray | tuple[jnp.ndarray, QMCState]:
        """
        Generate n samples from the Sobol sequence.

        Args:
            n (int): Number of samples to generate.
            state (QMCState | None): Optional explicit sampling state. When
                provided, returns ``(points, next_state)`` and does not mutate
                the engine's internal counter, so it can be used inside a
                compiled ``jax.lax.scan``. When ``None`` (default), returns
                only ``points`` and advances the engine's counter.
        Returns:
            jnp.ndarray: An array of shape (n, d) containing the Sobol
            sequence samples, or ``(points, next_state)`` when ``state`` is
            given.
        """
        if not isinstance(n, int):
            raise TypeError("n must be a Python integer.")

        if n <= 0:
            raise ValueError("Number of samples n must be positive.")

        if state is not None:
            return self._sample_from_state(n, state)

        # Eager path: validate bounds before generating.
        # Keep aligned power-of-two blocks for balanced-net experiments.
        # Dropping index zero can destroy balance (Owen, 2020).
        # With scrambling, the first point is generally not the origin.
        # Handle inverse-CDF endpoints without dropping an entire point.
        first_index = self.start_index + self._num_generated
        stop_index = first_index + n

        if stop_index > MAX_POINTS:
            raise ValueError(f"Requested points exceed maximum number of Sobol points ({MAX_POINTS}). Consider reducing n.")
        points, _ = self._sample_from_state(n, self._eager_state())
        self._num_generated += n
        return points

    def _validate_balanced_block(self, n: int) -> None:
        """Validate that ``n`` points form a balanced power-of-two block.

        A balanced Sobol' block requires a power-of-two count ``n`` and a start
        index divisible by ``n``. This is only required where a benchmark
        claims balanced-net sampling; ordinary sequence requests are not
        restricted.
        """
        if n <= 0 or n & (n - 1):
            raise ValueError("Balanced Sobol blocks require a power-of-two count.")
        first_index = self.start_index + self._num_generated
        if first_index % n:
            raise ValueError("Balanced Sobol blocks require an aligned start.")

    def reset(self):
        """Reset the Sobol sequence generator to its initial state."""
        self._num_generated = 0
        return self
    
@jax.jit
def _apply_lms(direction_integers: jnp.ndarray, lms_row_masks: jnp.ndarray) -> jnp.ndarray:
    """Apply LMS matrices to Sobol direction integers."""
    intersections = (direction_integers[:, :, None] & lms_row_masks[:, None, :])
    scrambled_bits = (jax.lax.population_count(intersections) & jnp.uint32(1))

    #
    bit_positions = jnp.arange(_MAXBITS -1, -1, -1, dtype=jnp.uint32)
    bit_weights = (jnp.uint32(1) << bit_positions)
    scrambled_direction_integers = jnp.sum(scrambled_bits * bit_weights[None, None, :], axis=-1, dtype=jnp.uint32)
    return scrambled_direction_integers
    
@partial(
    jax.jit,
    static_argnames=("n", "num_bits", "dtype"),
)
def _sobol_sample_batched(
    first_index: int,
    n: int,
    direction_integers: jax.Array,
    digital_shift: jax.Array,
    num_bits: int,
    dtype: jnp.dtype,
) -> jax.Array:
    """
    Generate consecutive Sobol points using a parallel XOR prefix scan.
    """
    first_indexes = jnp.asarray(first_index, dtype=jnp.uint32)

    # Calculate the first point in the batch directly from its Gray code.
    first_gray = first_indexes ^ (first_indexes >> jnp.uint32(1))

    bit_indices = jnp.arange(num_bits, dtype=jnp.uint32)

    first_gray_bits = ((first_gray >> bit_indices) & jnp.uint32(1)).astype(bool)

    # Shape of selected directions: (d, num_bits).
    first_contributions = jnp.where(first_gray_bits[None, :], direction_integers, jnp.uint32(0))

    first_point = jnp.bitwise_xor.reduce(first_contributions, axis=-1)

    # Apply the digital shift once. The recurrence then carries it
    # through the entire batch.
    first_point = first_point ^ digital_shift

    # Absolute indices of all remaining points.
    later_indices = (first_indexes + jnp.arange(1, n, dtype=jnp.uint32))

    # For a nonzero uint32 x:
    #
    #     x & -x
    #
    # isolates the least-significant set bit. Subtracting one creates
    # an integer containing ctz(x) set bits, so population_count gives
    # the trailing-zero count.
    lowest_set_bits = (later_indices & (jnp.uint32(0) - later_indices))

    direction_indices = jax.lax.population_count(lowest_set_bits - jnp.uint32(1))

    # direction_integers[:, direction_indices] has shape
    # (d, n - 1), so transpose it to (n - 1, d).
    recurrence_changes = jnp.take(direction_integers, direction_indices,axis=1,).T

    # The first entry is an absolute point. Every later entry is the
    # XOR change needed to move from the preceding point to the next.
    xor_inputs = jnp.concatenate((first_point[None, :], recurrence_changes,), axis=0)

    # Parallel prefix XOR:
    #
    # output[0] = first_point
    # output[1] = first_point XOR change[1]
    # output[2] = first_point XOR change[1] XOR change[2]
    # ...
    integer_points = jax.lax.associative_scan(jnp.bitwise_xor, xor_inputs, axis=0,)

    scale = jnp.asarray(2.0**(-num_bits), dtype=dtype,)

    points = integer_points.astype(dtype) * scale

    # Preserve the generator's [0, 1) contract. A 30-bit coordinate can round
    # up to exactly 1.0 in float32 (e.g. index 715827882), which would make a
    # normal inverse CDF infinite. Cap any rounded-one value at nextafter(1, 0)
    # in the requested dtype. Exact zero remains valid. Note that float32
    # cannot represent every distinct 30-bit coordinate, so clipping prevents
    # an invalid endpoint but does not restore lost precision; use float64 for
    # the principal comparison.
    zero = jnp.asarray(0, dtype=dtype)
    one = jnp.asarray(1, dtype=dtype)
    return jnp.minimum(points, jnp.nextafter(one, zero))

    # NOTE: jax.lax.scan iterates over 30 step.
    # def _sobol_sample_batched(
    #         self,
    #         first_index: int,
    #         n: int,
    #         direction_integers: jnp.ndarray,
    #         digital_shift: jnp.ndarray,
    #         num_bits: int,
    #         dtype: jnp.dtype,
    #     ) -> jnp.ndarray:
    #     """Generate consecutive batches of Sobol points. direction_integers (d, num_bits) and digital_shift (d,) are assumed to be precomputed for the desired dimension d."""
    #     indices = (jnp.arange(n, dtype=jnp.uint32) + jnp.uint32(first_index, dtype=jnp.uint32))

    #     # Gray-code representation of indices. This is used to determine which direction integers to XOR for each point. The operation consist of XORing the direction integers corresponding to the set bits in the Gray code of the index.
    #     gray_codes = indices ^ (indices >> jnp.uint32(1))

    #     initial_values = jnp.broadcast_to(digital_shift[None, :], (n, direction_integers.shape[0]))

    #     def add_direction_number(values, bit_index):
    #         gray_bit_is_set = ((gray_codes >> bit_index) & jnp.uint32(1)).astype(bool)
    #         direction = direction_integers[:, bit_index]
    #         values = jnp.where(gray_bit_is_set[:, None], values ^ direction[None, :], values)
    #         return values, None
    #     integer_points, _ = jax.lax.scan(add_direction_number, initial_values, jnp.arange(num_bits, dtype=jnp.uint32))
    #     scale = jnp.asarray(2.0 ** (-num_bits), dtype=dtype)
    #     return integer_points.astype(dtype) * scale


def normal_coordinates(u: jax.Array) -> jax.Array:
    """Standard-normal quantile with an explicit open-interval policy.

    Applies the 30-bit Sobol' boundary policy before the Gaussian inverse CDF:
    coordinates are clipped to ``[2^-31, 1 - 2^-31]`` (and additionally to
    ``nextafter(1, 0)`` in the requested dtype). This keeps the normal quantile
    finite for exact zero and rounded-one inputs, at the cost of a documented
    finite-precision approximation. Interior values are unchanged.

    Use this for initialization and propagation coordinates in both SMC and
    SQMC. Resampling coordinates need the separate cumulative-probability
    convention (see ``resample_from_uniform``).
    """
    zero = jnp.asarray(0, dtype=u.dtype)
    one = jnp.asarray(1, dtype=u.dtype)
    lower = jnp.asarray(2.0**-31, dtype=u.dtype)
    upper = jnp.minimum(
        jnp.asarray(1.0 - 2.0**-31, dtype=u.dtype),
        jnp.nextafter(one, zero),
    )
    return jax.scipy.special.ndtri(jnp.clip(u, lower, upper))


def main():
    from scipy.stats import qmc

    np.testing.assert_allclose(
        Halton(
            5,
            scramble=False,
            start_index=0,
            dtype=jnp.float64,
        ).sample(100),
        qmc.Halton(
            5,
            scramble=False,
        ).random(100),
        atol=1e-15,
        rtol=0.0,
    )
if __name__ == "__main__":
    main()