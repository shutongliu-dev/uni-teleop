"""Metric Hex geometry shared by scene authoring and private adjudication.

Dimensions are simulation design priors, not SHARPA reach/grasp qualification.
The piece body origin is the centre of the circular base, NOT the assembly COM.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

Cell = tuple[int, int]
SIZE = 7
COLORS = ('red', 'blue')
NEIGHBOR_OFFSETS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))


def validate_cell(q: int, r: int) -> Cell:
    if type(q) is not int or type(r) is not int or not (0 <= q < SIZE and 0 <= r < SIZE):
        raise ValueError('Hex coordinates must be integer q,r in [0,6]')
    return q, r


def neighbors(q: int, r: int) -> tuple[Cell, ...]:
    validate_cell(q, r)
    return tuple((q+dq, r+dr) for dq, dr in NEIGHBOR_OFFSETS
                 if 0 <= q+dq < SIZE and 0 <= r+dr < SIZE)


@dataclass(frozen=True)
class HexGeometry:
    pitch_m: float = .075
    center_xy_m: tuple[float, float] = (.65, 0.)
    board_yaw_rad: float = 0.
    table_top_z_m: float = .780
    board_thickness_m: float = .012
    board_margin_m: float = .020
    piece_radius_m: float = .028
    piece_base_height_m: float = .014
    handle_radius_m: float = .020
    handle_height_m: float = .040
    piece_mass_kg: float = .060
    reserve_pitch_m: float = .067
    reserve_inner_x_m: float = .234
    reserve_row_y_m: tuple[float, float, float] = (.300, .3675, .435)
    # Optional world-space positions: red_00..24 followed by blue_00..24.
    reserve_positions_m: tuple[tuple[float, float, float], ...] | None = None

    def __post_init__(self):
        def finite(value):
            if value is None:
                return True
            if isinstance(value, (tuple, list)):
                return all(finite(v) for v in value)
            return isinstance(value, (int, float)) and math.isfinite(value)

        for key, value in asdict(self).items():
            if not finite(value):
                raise ValueError(f'Nonfinite geometry: {key}')
        if any(getattr(self, key) <= 0 for key in (
                'pitch_m', 'board_thickness_m', 'board_margin_m', 'piece_radius_m',
                'piece_base_height_m', 'handle_radius_m', 'handle_height_m',
                'piece_mass_kg', 'reserve_pitch_m', 'reserve_inner_x_m')):
            raise ValueError('Geometry dimensions and mass must be positive')
        if len(self.center_xy_m) != 2 or len(self.reserve_row_y_m) != 3:
            raise ValueError('Geometry requires two XY coordinates and three reserve rows')
        if 2*self.piece_radius_m >= self.pitch_m or self.handle_radius_m > self.piece_radius_m:
            raise ValueError('Piece must fit its cell and handle must fit the base')
        if self.reserve_pitch_m <= 2*self.piece_radius_m:
            raise ValueError('Reserve pieces overlap')
        if self.reserve_positions_m is not None and (len(self.reserve_positions_m) != 50
                or any(len(position) != 3 for position in self.reserve_positions_m)):
            raise ValueError('Explicit reserve positions must provide all 50 world XYZ vectors')

    @property
    def board_top_z_m(self) -> float:
        return self.table_top_z_m + self.board_thickness_m

    @property
    def board_size_xy_m(self) -> tuple[float, float]:
        return (10*self.pitch_m + 2*self.board_margin_m,
                (3*math.sqrt(3)+2/math.sqrt(3))*self.pitch_m + 2*self.board_margin_m)

    @property
    def piece_height_m(self) -> float:
        return self.piece_base_height_m + self.handle_height_m

    def local_to_world_xy(self, x: float, y: float) -> tuple[float, float]:
        c, s = math.cos(self.board_yaw_rad), math.sin(self.board_yaw_rad)
        return self.center_xy_m[0]+c*x-s*y, self.center_xy_m[1]+s*x+c*y

    def world_to_local_xy(self, x: float, y: float) -> tuple[float, float]:
        c, s = math.cos(self.board_yaw_rad), math.sin(self.board_yaw_rad)
        dx, dy = x-self.center_xy_m[0], y-self.center_xy_m[1]
        return c*dx+s*dy, -s*dx+c*dy

    def cell_local_xy(self, q: int, r: int) -> tuple[float, float]:
        validate_cell(q, r)
        return self.pitch_m*((q-3)+.5*(r-3)), self.pitch_m*math.sqrt(3)/2*(r-3)

    def cell_center(self, q: int, r: int) -> tuple[float, float, float]:
        """Desired piece base-centre world pose, with upright local +Z."""
        x, y = self.local_to_world_xy(*self.cell_local_xy(q, r))
        return x, y, self.board_top_z_m+self.piece_base_height_m/2

    def cell_vertices(self, q: int, r: int) -> tuple[tuple[float, float], ...]:
        x, y = self.cell_local_xy(q, r)
        radius = self.pitch_m/math.sqrt(3)
        return tuple(self.local_to_world_xy(x+radius*math.cos(math.pi/6+k*math.pi/3),
                                          y+radius*math.sin(math.pi/6+k*math.pi/3))
                     for k in range(6))

    def footprint_clearance_m(self, q: int, r: int, position_xy: tuple[float, float],
                              radius_m: float) -> float:
        """Smallest clearance to six cell edges, for a conservative circular footprint."""
        x, y = self.world_to_local_xy(*position_xy)
        cx, cy = self.cell_local_xy(q, r)
        return min(self.pitch_m/2 - ((x-cx)*math.cos(k*math.pi/3)
                                    +(y-cy)*math.sin(k*math.pi/3)) - radius_m
                   for k in range(6))

    def reserve_position(self, color: str, index: int) -> tuple[float, float, float]:
        if color not in COLORS or type(index) is not int or not 0 <= index < 25:
            raise ValueError('Reserve identity requires red/blue and index 0..24')
        if self.reserve_positions_m is not None:
            return self.reserve_positions_m[COLORS.index(color)*25+index]
        # Three rows of eight avoid the central near-base region; one extra
        # token sits at the centre of the innermost row. This is still subject
        # to native collision/contact qualification after kinematic checks.
        row, column = divmod(index, 8)
        if index == 24:
            row, x = 0, 0.
        elif column < 4:
            x = -(self.reserve_inner_x_m+(3-column)*self.reserve_pitch_m)
        else:
            x = self.reserve_inner_x_m+(column-4)*self.reserve_pitch_m
        y = self.reserve_row_y_m[row] * (-1 if color == 'red' else 1)
        # Reserve rows follow the board frame, independently of robot identities.
        wx, wy = self.local_to_world_xy(x, y)
        return wx, wy, self.table_top_z_m+self.piece_base_height_m/2

    @staticmethod
    def cell_label(q: int, r: int) -> str:
        validate_cell(q, r)
        return chr(ord('A')+q)+str(r+1)

    def specification(self) -> dict:
        return {**asdict(self), 'size': SIZE, 'board_size_xy_m': self.board_size_xy_m,
                'board_top_z_m': self.board_top_z_m, 'piece_height_m': self.piece_height_m,
                'piece_origin': 'base_centre', 'quaternion_convention': 'wxyz',
                'axes': 'q:+local_x; r:+local_x/2,+sqrt(3)*local_y/2',
                'dimensions_source': 'versioned_simulation_design_prior_not_measured_hardware',
                'handle_purpose': '40mm upright pinch surface above the 56mm base; native grasp unqualified',
                'piece_count_per_color': 25}
