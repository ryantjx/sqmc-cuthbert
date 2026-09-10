"""Generate the packed Sobol direction-integer table used by ``qmc.py``.

This is an offline build utility. It parses the Joe--Kuo D(6) source table,
constructs the primitive-polynomial and initial-direction arrays used by
SciPy, expands them with the Bratley--Fox recurrence, and writes the final
bit-scaled direction integers to ``sobol_data.npz``.

The generated file is a runtime data artifact; this module should not be
imported by ``qmc.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


_MAX_DIMENSION = 21_201
_MAX_DEGREE = 18
_DEFAULT_BITS = 30
_HERE = Path(__file__).resolve().parent
_DEFAULT_SOURCE = _HERE / "new-joe-kuo-6.21201"
_DEFAULT_OUTPUT = _HERE / "_sobol_direction_numbers.npz"


def _parse_joe_kuo(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse Joe--Kuo data into SciPy-style ``poly`` and ``vinit`` arrays."""
    poly = np.zeros(_MAX_DIMENSION, dtype=np.uint32)
    vinit = np.zeros((_MAX_DIMENSION, _MAX_DEGREE), dtype=np.uint32)

    # Dimension one is not present in the Joe--Kuo text file.
    poly[0] = np.uint32(1)
    vinit[0, 0] = np.uint32(1)

    with path.open("r", encoding="utf-8") as source:
        next(source)  # Skip: d, s, a, m_i.

        expected_dimension = 2
        for line_number, line in enumerate(source, start=2):
            fields = line.split()
            if not fields:
                continue

            values = [int(field) for field in fields]
            dimension, degree, coefficient = values[:3]
            initial_values = values[3:]

            if dimension != expected_dimension:
                raise ValueError(
                    f"{path}:{line_number}: expected dimension "
                    f"{expected_dimension}, found {dimension}."
                )
            if not 1 <= degree <= _MAX_DEGREE:
                raise ValueError(
                    f"{path}:{line_number}: polynomial degree {degree} "
                    f"is outside [1, {_MAX_DEGREE}]."
                )
            if len(initial_values) != degree:
                raise ValueError(
                    f"{path}:{line_number}: degree {degree} requires "
                    f"{degree} initial values, found {len(initial_values)}."
                )

            for index, initial_value in enumerate(initial_values):
                upper_bound = 1 << (index + 1)
                if initial_value <= 0 or initial_value >= upper_bound:
                    raise ValueError(
                        f"{path}:{line_number}: m_{index + 1} must be in "
                        f"[1, {upper_bound}), found {initial_value}."
                    )
                if initial_value % 2 == 0:
                    raise ValueError(
                        f"{path}:{line_number}: m_{index + 1} must be odd, "
                        f"found {initial_value}."
                    )

            row = dimension - 1

            # Restore the leading and constant coefficients omitted from a.
            poly[row] = np.uint32(
                (1 << degree) | (coefficient << 1) | 1
            )
            vinit[row, :degree] = np.asarray(
                initial_values,
                dtype=np.uint32,
            )
            expected_dimension += 1

    if expected_dimension != _MAX_DIMENSION + 1:
        raise ValueError(
            f"{path}: expected {_MAX_DIMENSION - 1} data rows, found "
            f"{expected_dimension - 2}."
        )

    return poly, vinit


def _initialize_direction_integers(
    poly: np.ndarray,
    vinit: np.ndarray,
    bits: int,
) -> np.ndarray:
    """Expand ``poly`` and ``vinit`` using SciPy's Sobol recurrence."""
    if not 1 <= bits <= 64:
        raise ValueError("bits must be in [1, 64].")

    integer_dtype = np.uint32 if bits <= 32 else np.uint64
    dimension = poly.shape[0]
    directions = np.zeros((dimension, bits), dtype=integer_dtype)

    # Unscaled direction numerators for dimension one are all one.
    directions[0, :] = integer_dtype(1)

    for dim in range(1, dimension):
        polynomial = int(poly[dim])
        degree = polynomial.bit_length() - 1
        directions[dim, :degree] = vinit[dim, :degree]

        for column in range(degree, bits):
            new_value = int(directions[dim, column - degree])
            power_of_two = 1

            for coefficient_index in range(degree):
                power_of_two <<= 1
                polynomial_bit = (
                    polynomial >> (degree - 1 - coefficient_index)
                ) & 1

                if polynomial_bit:
                    new_value ^= power_of_two * int(
                        directions[dim, column - coefficient_index - 1]
                    )

            directions[dim, column] = integer_dtype(new_value)

    column_scales = np.asarray(
        [1 << (bits - 1 - column) for column in range(bits)],
        dtype=integer_dtype,
    )
    directions *= column_scales[None, :]

    return directions


def _validate_direction_integers(
    direction_integers: np.ndarray,
    bits: int,
) -> None:
    expected_dtype = np.dtype(np.uint32 if bits <= 32 else np.uint64)
    expected_shape = (_MAX_DIMENSION, bits)

    if direction_integers.shape != expected_shape:
        raise ValueError(
            f"Expected direction-integer shape {expected_shape}, found "
            f"{direction_integers.shape}."
        )
    if direction_integers.dtype != expected_dtype:
        raise ValueError(
            f"Expected direction-integer dtype {expected_dtype}, found "
            f"{direction_integers.dtype}."
        )
    if np.any(direction_integers == 0):
        raise ValueError("Direction integers must all be nonzero.")


def _verify_against_scipy(direction_integers: np.ndarray, bits: int) -> None:
    """Compare with SciPy's private table as an optional build-time check."""
    try:
        from scipy.stats import qmc
    except ImportError as error:
        raise RuntimeError(
            "--verify-scipy requires SciPy to be installed."
        ) from error

    reference = qmc.Sobol(
        d=direction_integers.shape[0],
        scramble=False,
        bits=bits,
    )._sv
    np.testing.assert_array_equal(direction_integers, reference)


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate packed Joe--Kuo Sobol direction integers."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=_DEFAULT_SOURCE,
        help=f"Joe--Kuo source table (default: {_DEFAULT_SOURCE}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Generated NPZ file (default: {_DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=_DEFAULT_BITS,
        help=f"Number of direction bits (default: {_DEFAULT_BITS}).",
    )
    parser.add_argument(
        "--verify-scipy",
        action="store_true",
        help="Verify the generated table against SciPy's private _sv table.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the output file if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_arguments()

    if arguments.output.exists() and not arguments.force:
        raise FileExistsError(
            f"{arguments.output} already exists; pass --force to replace it."
        )

    poly, vinit = _parse_joe_kuo(arguments.source)
    direction_integers = _initialize_direction_integers(
        poly=poly,
        vinit=vinit,
        bits=arguments.bits,
    )
    _validate_direction_integers(direction_integers, arguments.bits)

    if arguments.verify_scipy:
        _verify_against_scipy(direction_integers, arguments.bits)
        print("Generated direction integers match SciPy exactly.")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arguments.output,
        direction_integers=direction_integers,
    )
    print(
        f"Wrote {direction_integers.shape} {direction_integers.dtype} "
        f"direction integers to {arguments.output}."
    )


if __name__ == "__main__":
    main()
