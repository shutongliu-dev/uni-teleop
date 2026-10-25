"""Sparse garment-grid spreading for tactile normal loads."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class GarmentLoadSpreadKernel:
    """Spread taxel loads over compact garment-grid neighborhoods.

    Each input taxel keeps the identity fraction and shares fixed total
    fractions with its graph-distance-one and graph-distance-two neighbors.
    Columns are normalized independently, so spreading preserves total load.
    Grid edges never connect different ``(device_id, patch_id)`` pairs.
    """

    def __init__(
        self,
        device_id: Sequence[int] | np.ndarray,
        patch_id: Sequence[int] | np.ndarray,
        grid_row: Sequence[int] | np.ndarray,
        grid_col: Sequence[int] | np.ndarray,
        wrap_columns: bool | Sequence[bool] | np.ndarray,
        *,
        first_ring_fraction: float = 0.0,
        second_ring_fraction: float = 0.0,
    ) -> None:
        self.device_id = _integer_vector(device_id, "device_id")
        self.patch_id = _integer_vector(patch_id, "patch_id")
        self.grid_row = _integer_vector(grid_row, "grid_row")
        self.grid_col = _integer_vector(grid_col, "grid_col")
        self.size = len(self.device_id)
        if self.size == 0:
            raise ValueError("garment spread kernel needs at least one taxel")
        for name, value in (
            ("patch_id", self.patch_id),
            ("grid_row", self.grid_row),
            ("grid_col", self.grid_col),
        ):
            if len(value) != self.size:
                raise ValueError(f"{name} must match device_id length")

        self.wrap_columns = _boolean_vector(wrap_columns, self.size)
        self.first_ring_fraction = _fraction(
            first_ring_fraction,
            "first_ring_fraction",
        )
        self.second_ring_fraction = _fraction(
            second_ring_fraction,
            "second_ring_fraction",
        )
        if self.first_ring_fraction + self.second_ring_fraction > 1.0:
            raise ValueError("garment ring fractions must sum to at most one")

        adjacency = self._build_adjacency()
        self.column_ptr, self.output_index, self.weight = self._build_sparse(adjacency)
        self.input_index = np.repeat(
            np.arange(self.size, dtype=np.int32),
            np.diff(self.column_ptr),
        )
        self.input_index.setflags(write=False)
        self._identity = (
            self.first_ring_fraction == 0.0 and self.second_ring_fraction == 0.0
        )

    @property
    def column_sums(self) -> np.ndarray:
        """Return one load-conservation sum per input taxel."""

        return np.add.reduceat(self.weight, self.column_ptr[:-1])

    def spread(self, normal_force: Sequence[float] | np.ndarray) -> np.ndarray:
        """Return a deterministic, nonnegative load-spread copy."""

        force = np.asarray(normal_force)
        if force.shape != (self.size,):
            raise ValueError(f"normal_force must have shape ({self.size},)")
        if not (
            np.issubdtype(force.dtype, np.integer)
            or np.issubdtype(force.dtype, np.floating)
        ):
            raise TypeError("normal_force must be numeric")
        if np.any(~np.isfinite(force)) or np.any(force < 0):
            raise ValueError("normal_force must be finite and nonnegative")
        if self._identity:
            return force.copy()

        contributions = self.weight * force[self.input_index]
        result = np.bincount(
            self.output_index,
            weights=contributions,
            minlength=self.size,
        )
        return result.astype(np.result_type(force.dtype, np.float64), copy=False)

    def _build_adjacency(self) -> tuple[frozenset[int], ...]:
        neighbors = [set() for _ in range(self.size)]
        groups: dict[tuple[int, int], list[int]] = {}
        for index, key in enumerate(zip(self.device_id, self.patch_id, strict=True)):
            groups.setdefault((int(key[0]), int(key[1])), []).append(index)

        for indices in groups.values():
            wraps = self.wrap_columns[indices]
            if np.any(wraps != wraps[0]):
                raise ValueError("wrap_columns must be constant within each patch")
            coordinates: dict[tuple[int, int], int] = {}
            for index in indices:
                coordinate = int(self.grid_row[index]), int(self.grid_col[index])
                if coordinate in coordinates:
                    raise ValueError(
                        "garment grid coordinates must be unique within each patch"
                    )
                coordinates[coordinate] = index

            columns = sorted({column for _, column in coordinates})
            minimum_column, maximum_column = columns[0], columns[-1]
            for (row, column), index in coordinates.items():
                candidates = (
                    (row - 1, column),
                    (row + 1, column),
                    (row, column - 1),
                    (row, column + 1),
                )
                for coordinate in candidates:
                    neighbor = coordinates.get(coordinate)
                    if neighbor is not None:
                        neighbors[index].add(neighbor)
                if bool(wraps[0]) and minimum_column != maximum_column:
                    seam_column = None
                    if column == minimum_column:
                        seam_column = maximum_column
                    elif column == maximum_column:
                        seam_column = minimum_column
                    if seam_column is not None:
                        neighbor = coordinates.get((row, seam_column))
                        if neighbor is not None:
                            neighbors[index].add(neighbor)

        return tuple(frozenset(values) for values in neighbors)

    def _build_sparse(
        self,
        adjacency: tuple[frozenset[int], ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        column_ptr = [0]
        output_index: list[int] = []
        weight: list[float] = []
        for source, first_ring_values in enumerate(adjacency):
            first_ring = sorted(first_ring_values)
            second_ring_values: set[int] = set()
            for neighbor in first_ring:
                second_ring_values.update(adjacency[neighbor])
            second_ring_values.discard(source)
            second_ring_values.difference_update(first_ring)
            second_ring = sorted(second_ring_values)

            entries: dict[int, float] = {}
            first_fraction = self.first_ring_fraction if first_ring else 0.0
            second_fraction = self.second_ring_fraction if second_ring else 0.0
            entries[source] = 1.0 - first_fraction - second_fraction
            if first_ring and first_fraction > 0.0:
                share = first_fraction / len(first_ring)
                entries.update((index, share) for index in first_ring)
            if second_ring and second_fraction > 0.0:
                share = second_fraction / len(second_ring)
                entries.update((index, share) for index in second_ring)

            ordered = sorted(entries.items())
            correction = 1.0 - sum(value for _, value in ordered)
            entries[source] += correction
            ordered = sorted(
                (index, value) for index, value in entries.items() if value > 0.0
            )
            output_index.extend(index for index, _ in ordered)
            weight.extend(value for _, value in ordered)
            column_ptr.append(len(output_index))

        arrays = (
            np.asarray(column_ptr, dtype=np.int32),
            np.asarray(output_index, dtype=np.int32),
            np.asarray(weight, dtype=np.float64),
        )
        for value in arrays:
            value.setflags(write=False)
        return arrays


def _integer_vector(value: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape (N,)")
    if np.issubdtype(array.dtype, np.bool_) or not np.issubdtype(
        array.dtype, np.integer
    ):
        raise TypeError(f"{name} must contain integers")
    result = np.asarray(array, dtype=np.int64).copy()
    result.setflags(write=False)
    return result


def _boolean_vector(
    value: bool | Sequence[bool] | np.ndarray,
    size: int,
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 0:
        if not np.issubdtype(array.dtype, np.bool_):
            raise TypeError("wrap_columns must contain booleans")
        array = np.full(size, bool(array), dtype=bool)
    if array.shape != (size,):
        raise ValueError(f"wrap_columns must be scalar or have shape ({size},)")
    if not np.issubdtype(array.dtype, np.bool_):
        raise TypeError("wrap_columns must contain booleans")
    result = np.asarray(array, dtype=bool).copy()
    result.setflags(write=False)
    return result


def _fraction(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return result
