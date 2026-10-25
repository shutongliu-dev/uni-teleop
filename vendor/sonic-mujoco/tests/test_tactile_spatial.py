import unittest

import numpy as np

from sonic_mujoco.tactile_spatial import GarmentLoadSpreadKernel


def _grid(
    rows: int,
    columns: int,
    *,
    device: int = 0,
    patch: int = 0,
    wrap: bool = False,
) -> GarmentLoadSpreadKernel:
    row, column = np.indices((rows, columns))
    size = rows * columns
    return GarmentLoadSpreadKernel(
        np.full(size, device),
        np.full(size, patch),
        row.ravel(),
        column.ravel(),
        wrap,
        first_ring_fraction=0.4,
        second_ring_fraction=0.2,
    )


class GarmentLoadSpreadKernelTest(unittest.TestCase):
    def test_nonnegative_sparse_kernel_conserves_every_input_column(self) -> None:
        kernel = _grid(4, 5)
        force = np.random.default_rng(12).uniform(0.0, 30.0, kernel.size)
        original = force.copy()

        spread = kernel.spread(force)
        repeated = kernel.spread(force)

        np.testing.assert_allclose(kernel.column_sums, 1.0, atol=1e-15)
        self.assertTrue(np.all(kernel.weight >= 0.0))
        self.assertTrue(np.all(spread >= 0.0))
        self.assertAlmostEqual(float(np.sum(spread)), float(np.sum(force)), places=12)
        np.testing.assert_array_equal(spread, repeated)
        np.testing.assert_array_equal(force, original)

        reference = np.zeros(kernel.size)
        np.add.at(
            reference,
            kernel.output_index,
            kernel.weight * force[kernel.input_index],
        )
        np.testing.assert_array_equal(spread, reference)

    def test_single_point_has_identity_first_and_second_ring_load(self) -> None:
        kernel = _grid(5, 5)
        force = np.zeros(kernel.size)
        force[2 * 5 + 2] = 10.0

        spread = kernel.spread(force).reshape(5, 5)

        self.assertAlmostEqual(spread[2, 2], 4.0)
        first = np.array((spread[1, 2], spread[2, 1], spread[2, 3], spread[3, 2]))
        np.testing.assert_allclose(first, 1.0)
        second_mask = np.fromfunction(
            lambda row, column: abs(row - 2) + abs(column - 2) == 2,
            (5, 5),
        )
        np.testing.assert_allclose(spread[second_mask], 0.25)
        self.assertAlmostEqual(float(np.sum(spread[second_mask])), 2.0)
        expected = np.zeros((5, 5))
        expected[2, 2] = 4.0
        expected[1, 2] = expected[2, 1] = 1.0
        expected[2, 3] = expected[3, 2] = 1.0
        expected[second_mask] = 0.25
        np.testing.assert_allclose(spread, expected)

    def test_wrapped_columns_connect_the_sleeve_seam(self) -> None:
        wrapped = _grid(3, 4, wrap=True)
        unwrapped = _grid(3, 4, wrap=False)
        force = np.zeros(12)
        force[4] = 10.0  # row 1, column 0

        wrapped_value = wrapped.spread(force).reshape(3, 4)
        unwrapped_value = unwrapped.spread(force).reshape(3, 4)

        self.assertGreater(wrapped_value[1, 3], 0.0)
        self.assertEqual(unwrapped_value[1, 3], 0.0)

    def test_devices_and_patches_are_strictly_isolated(self) -> None:
        # Four identical 1x3 grids distinguished only by device and patch.
        device = np.repeat((0, 0, 1, 1), 3)
        patch = np.repeat((0, 1, 0, 1), 3)
        row = np.zeros(12, dtype=int)
        column = np.tile(np.arange(3), 4)
        kernel = GarmentLoadSpreadKernel(
            device,
            patch,
            row,
            column,
            False,
            first_ring_fraction=0.5,
            second_ring_fraction=0.25,
        )
        force = np.zeros(12)
        force[1] = 8.0

        spread = kernel.spread(force)

        self.assertAlmostEqual(float(np.sum(spread[:3])), 8.0)
        np.testing.assert_array_equal(spread[3:], 0.0)

    def test_zero_kernel_is_bitwise_identity(self) -> None:
        row, column = np.indices((3, 4))
        kernel = GarmentLoadSpreadKernel(
            np.zeros(12, dtype=int),
            np.zeros(12, dtype=int),
            row.ravel(),
            column.ravel(),
            True,
        )
        force = np.array(
            [-0.0, 0.0, 1.0, 3.25, 8.5, 13.0, 21.0, 0.0, 2.0, 5.0, 1.5, 9.0],
            dtype=np.float32,
        )

        spread = kernel.spread(force)

        self.assertEqual(spread.dtype, force.dtype)
        self.assertEqual(spread.tobytes(), force.tobytes())
        self.assertFalse(np.shares_memory(spread, force))
        self.assertEqual(len(kernel.weight), kernel.size)

    def test_rejects_ambiguous_topology_and_invalid_fraction(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            GarmentLoadSpreadKernel([0, 0], [0, 0], [0, 0], [0, 0], False)
        with self.assertRaisesRegex(ValueError, "constant"):
            GarmentLoadSpreadKernel([0, 0], [0, 0], [0, 0], [0, 1], [False, True])
        with self.assertRaisesRegex(ValueError, "sum"):
            GarmentLoadSpreadKernel(
                [0],
                [0],
                [0],
                [0],
                False,
                first_ring_fraction=0.7,
                second_ring_fraction=0.4,
            )


if __name__ == "__main__":
    unittest.main()
