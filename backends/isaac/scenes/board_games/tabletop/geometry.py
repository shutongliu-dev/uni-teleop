"""Metric board designs, independent of the fixed card-room environment.

Dimensions are versioned simulation design priors, not grasp qualification.
The token body origin is the centre of its base, not its assembly COM.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

Cell = tuple[int, int]
COLORS = ('black', 'white')
GAME_IDS = ('tic_tac_toe_3x3', 'connect_four_5x5', 'hex7x7')
NEIGHBOR_OFFSETS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1))
_DESIGNS = {'tic_tac_toe_3x3': (1, 3, 'square', 3, .140),
            'connect_four_5x5': (2, 5, 'square', 4, .115),
            'hex7x7': (3, 7, 'hex', None, .090)}


@dataclass(frozen=True)
class BoardGameGeometry:
    game_id: str = 'tic_tac_toe_3x3'
    level: int = 1
    size: int = 3
    shape: str = 'square'
    win_length: int | None = 3
    pitch_m: float = .140
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
    # Minimum centre separation, with actual columns/rows declared separately.
    reserve_pitch_m: float = .140
    reserve_columns_x_m: tuple[float, ...] = (.510,)
    reserve_row_y_m: tuple[float, ...] = (.380, .520, .660)
    reserve_surface_z_m: float = .800
    # Explicit world positions in black inventory order, then white order.
    reserve_positions_m: tuple[tuple[float, float, float], ...] | None = None

    def __post_init__(self):
        design = _DESIGNS.get(self.game_id)
        if design is None or (self.level, self.size, self.shape, self.win_length) != design[:4]:
            raise ValueError('Use geometry_for(game_id) for one of the three supported games')
        if type(self.level) is not int or type(self.size) is not int:
            raise ValueError('Level and board size must be integers')
        def finite(value):
            if value is None or isinstance(value, str):
                return True
            if isinstance(value, (tuple, list)):
                return all(finite(v) for v in value)
            return type(value) in (int, float) and math.isfinite(value)
        for key, value in asdict(self).items():
            if not finite(value):
                raise ValueError('Nonfinite geometry: '+key)
        if any(getattr(self, key) <= 0 for key in (
                'pitch_m', 'board_thickness_m', 'board_margin_m', 'piece_radius_m',
                'piece_base_height_m', 'handle_radius_m', 'handle_height_m',
                'piece_mass_kg', 'reserve_pitch_m', 'reserve_surface_z_m')):
            raise ValueError('Geometry dimensions and mass must be positive')
        if (len(self.center_xy_m) != 2 or not self.reserve_columns_x_m or not self.reserve_row_y_m
                or any(v <= 0 for v in (*self.reserve_columns_x_m, *self.reserve_row_y_m))):
            raise ValueError('Geometry requires XY centre and positive lateral reserve columns/rows')
        if len(self.reserve_columns_x_m)*len(self.reserve_row_y_m) < (self.inventory_counts['black']+1)//2:
            raise ValueError('Reserve station has too few physical slots')
        if 2*self.piece_radius_m >= self.pitch_m or self.handle_radius_m > self.piece_radius_m:
            raise ValueError('Piece must fit its cell and handle must fit its base')
        if self.reserve_pitch_m <= 2*self.piece_radius_m:
            raise ValueError('Reserve pieces overlap')
        if self.reserve_positions_m is not None and (
                len(self.reserve_positions_m) != self.size**2
                or any(len(position) != 3 for position in self.reserve_positions_m)):
            raise ValueError('Explicit reserve positions must provide the complete inventory')

    @property
    def inventory_counts(self) -> dict[str, int]:
        return {'black': (self.size**2+1)//2, 'white': self.size**2//2}

    @property
    def board_top_z_m(self) -> float:
        return self.table_top_z_m+self.board_thickness_m

    @property
    def board_size_xy_m(self) -> tuple[float, float]:
        if self.shape == 'square':
            side = self.size*self.pitch_m+2*self.board_margin_m
            return side, side
        return ((1.5*(self.size-1)+1)*self.pitch_m+2*self.board_margin_m,
                ((self.size-1)*math.sqrt(3)/2+2/math.sqrt(3))*self.pitch_m+2*self.board_margin_m)

    @property
    def piece_height_m(self) -> float:
        return self.piece_base_height_m+self.handle_height_m

    def validate_cell(self, q: int, r: int) -> Cell:
        if type(q) is not int or type(r) is not int or not (0 <= q < self.size and 0 <= r < self.size):
            raise ValueError(f'Coordinates must be integer q,r in [0,{self.size-1}]')
        return q, r

    def neighbors(self, q: int, r: int) -> tuple[Cell, ...]:
        self.validate_cell(q, r)
        offsets = NEIGHBOR_OFFSETS if self.shape == 'hex' else NEIGHBOR_OFFSETS[:4]
        return tuple((q+dq, r+dr) for dq, dr in offsets
                     if 0 <= q+dq < self.size and 0 <= r+dr < self.size)

    def local_to_world_xy(self, x: float, y: float) -> tuple[float, float]:
        c, s = math.cos(self.board_yaw_rad), math.sin(self.board_yaw_rad)
        return self.center_xy_m[0]+c*x-s*y, self.center_xy_m[1]+s*x+c*y

    def world_to_local_xy(self, x: float, y: float) -> tuple[float, float]:
        c, s = math.cos(self.board_yaw_rad), math.sin(self.board_yaw_rad)
        dx, dy = x-self.center_xy_m[0], y-self.center_xy_m[1]
        return c*dx+s*dy, -s*dx+c*dy

    def cell_local_xy(self, q: int, r: int) -> tuple[float, float]:
        self.validate_cell(q, r)
        half = (self.size-1)/2
        if self.shape == 'square':
            return self.pitch_m*(q-half), self.pitch_m*(r-half)
        return self.pitch_m*((q-half)+.5*(r-half)), self.pitch_m*math.sqrt(3)/2*(r-half)

    def cell_center(self, q: int, r: int) -> tuple[float, float, float]:
        x, y = self.local_to_world_xy(*self.cell_local_xy(q, r))
        return x, y, self.board_top_z_m+self.piece_base_height_m/2

    def cell_vertices(self, q: int, r: int) -> tuple[tuple[float, float], ...]:
        x, y = self.cell_local_xy(q, r)
        if self.shape == 'square':
            half = self.pitch_m/2
            return tuple(self.local_to_world_xy(x+dx, y+dy)
                         for dx, dy in ((-half, -half), (half, -half), (half, half), (-half, half)))
        radius = self.pitch_m/math.sqrt(3)
        return tuple(self.local_to_world_xy(x+radius*math.cos(math.pi/6+k*math.pi/3),
                                           y+radius*math.sin(math.pi/6+k*math.pi/3))
                     for k in range(6))

    def footprint_clearance_m(self, q: int, r: int, position_xy: tuple[float, float],
                              radius_m: float) -> float:
        x, y = self.world_to_local_xy(*position_xy)
        cx, cy = self.cell_local_xy(q, r)
        if self.shape == 'square':
            return self.pitch_m/2-max(abs(x-cx), abs(y-cy))-radius_m
        return min(self.pitch_m/2-((x-cx)*math.cos(k*math.pi/3)
                                   +(y-cy)*math.sin(k*math.pi/3))-radius_m for k in range(6))

    def reserve_position(self, color: str, index: int) -> tuple[float, float, float]:
        if color not in COLORS or type(index) is not int or not 0 <= index < self.inventory_counts[color]:
            raise ValueError('Reserve identity requires black/white and a valid inventory index')
        if self.reserve_positions_m is not None:
            offset = 0 if color == 'black' else self.inventory_counts['black']
            return self.reserve_positions_m[offset+index]
        sign = -1 if color == 'black' else 1
        first = (self.inventory_counts[color]+1)//2
        wing, slot = (-1, index) if index < first else (1, index-first)
        row, column = divmod(slot, len(self.reserve_columns_x_m))
        # The odd 25th black token is centred on a fifth row, not squeezed
        # between its neighbours. All remaining Hex stations have 12 slots.
        if self.level == 3 and slot == 12:
            column = len(self.reserve_columns_x_m)//2
        x = wing*self.reserve_columns_x_m[column]
        y = sign*self.reserve_row_y_m[row]
        wx, wy = self.local_to_world_xy(x, y)
        return wx, wy, self.reserve_surface_z_m+self.piece_base_height_m/2

    def reserve_fixture_id(self, color: str, index: int) -> str:
        self.reserve_position(color, index)  # Validate identity and slot.
        wing = 'west' if index < (self.inventory_counts[color]+1)//2 else 'east'
        return f'reserve_{color}_{wing}'

    def reserve_stations(self) -> dict[str, tuple[tuple[float, float, float], ...]]:
        groups = {}
        for color, count in self.inventory_counts.items():
            for index in range(count):
                groups.setdefault(self.reserve_fixture_id(color, index), []).append(self.reserve_position(color, index))
        return {name: tuple(positions) for name, positions in groups.items()}

    def cell_label(self, q: int, r: int) -> str:
        self.validate_cell(q, r)
        return chr(ord('A')+q)+str(r+1)

    def specification(self) -> dict:
        return {**asdict(self), 'board_size_xy_m': self.board_size_xy_m,
                'board_top_z_m': self.board_top_z_m, 'piece_height_m': self.piece_height_m,
                'piece_origin': 'base_centre', 'quaternion_convention': 'wxyz',
                'axes': ('q:+local_x; r:+local_y' if self.shape == 'square' else
                         'q:+local_x; r:+local_x/2,+sqrt(3)*local_y/2'),
                'dimensions_source': 'versioned_simulation_design_prior_not_measured_hardware',
                'handle_purpose': '40mm upright pinch surface above the 56mm base; native grasp unqualified',
                'reserve_layout': 'four_lateral_supply_stations_v1',
                'reserve_pitch_scope': 'minimum designed centre separation, not grasp clearance',
                'inventory_counts': self.inventory_counts,
                'piece_count': self.size**2}


def geometry_for(game_id: str) -> BoardGameGeometry:
    if game_id not in _DESIGNS:
        raise ValueError('Unknown board game: '+str(game_id))
    level, size, shape, win_length, pitch = _DESIGNS[game_id]
    reserve_pitch, columns, rows = {
        1: (.140, (.510,), (.380, .520, .660)),
        2: (.120, (.590, .430), (.360, .480, .600, .720)),
        3: (.100, (.610, .510, .410), (.360, .460, .560, .660, .760)),
    }[level]
    return BoardGameGeometry(game_id=game_id, level=level, size=size, shape=shape,
                             win_length=win_length, pitch_m=pitch,
                             reserve_pitch_m=reserve_pitch, reserve_columns_x_m=columns,
                             reserve_row_y_m=rows)
