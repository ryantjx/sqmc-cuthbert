# Hilbert sorting (`sqmc/hilbert_sort`)

JAX implementation of Hilbert space-filling-curve sorting, used to order
particles before SQMC resampling.

## Overview

The module computes a packed 62-bit Hilbert index for each point and sorts the
points by that index. It is the standard Hilbert space-filling curve of the
reference `particles` routine and of the archived `hilbert_adrien` baseline:
each `d`-coordinate point is bit-transposed into traversal chunks, walked chunk
by chunk through the Hilbert cube, and packed into a single scalar index, which
then orders the points in `hilbert_sort`.

The implementation uses `uint64` bit-integer arithmetic throughout, which gives
a measured ~1.6× speedup over the `hilbert_adrien` baseline on a batched
100k-point, `d=3` sort. Double precision is enabled so that coordinate
quantisation is consistent with the 62-bit integer representation. It supports
`2 ≤ d ≤ 62` dimensions.

## Public API

- `hilbert_sort(x)` — returns a permutation of `arange(n)` ordering the points
  by Hilbert index. Standardises columns, applies a logistic transform,
  quantises to a power-of-two grid, computes packed 62-bit Hilbert indices, and
  argsorts.
- `Hilbert_to_int(coords, max_int)` — packed Hilbert index of a coordinate.
- `gray_encode` / `gray_decode` — Gray-code conversion.
- `gray_encode_travel` / `gray_decode_travel` — Gray-code conversion with
  traversal state.
- `child_start_end` / `initial_start_end` — Hilbert-cube traversal bounds.
- `transpose_bits` — vectorised bit-level transposition.
- `pack_index` / `unpack_coords` — pack/unpack coordinates to/from a scalar
  index.
- `invlogit` — logistic transform used before quantisation.

## Usage

```python
from sqmc.hilbert_sort.hilbert_sort import hilbert_sort

order = hilbert_sort(points)   # permutation of arange(n) by Hilbert index
```

## Tests

```bash
python -m pytest sqmc/tests/test_hilbert_sort.py -q
```

## References

See `HILBERT_SORT.md` in this directory for the particle-sorting, parallel
Hilbert, and fast-sort references.
