"""Pure Hex rules. Physical callers use HexMatch, never call play directly."""
from __future__ import annotations

from collections import deque
from types import MappingProxyType

from .geometry import COLORS, SIZE, Cell, neighbors, validate_cell


class IllegalMove(ValueError):
    pass


class HexRules:
    """Fixed red r-edge / blue q-edge goals; pie/swap is disabled in v1.

    A future competitive report must pair games with reversed color assignments.
    No claim of equal first-move advantage follows from equal observations.
    """

    def __init__(self, players: tuple[str, str] = ('player_0', 'player_1')):
        if len(players) != 2 or len(set(players)) != 2 or not all(isinstance(p, str) and p for p in players):
            raise ValueError('Exactly two distinct nonempty player IDs are required')
        self.players = tuple(players)
        self._board: dict[Cell, str] = {}
        self.turn_index = 0
        self.winner: str | None = None
        self.winning_path: tuple[Cell, ...] = ()

    @property
    def board(self):
        return MappingProxyType(self._board)

    @property
    def current_player(self) -> str | None:
        return None if self.winner else self.players[self.turn_index % 2]

    def color(self, player_id: str) -> str:
        if player_id not in self.players:
            raise IllegalMove('Unknown player')
        return COLORS[self.players.index(player_id)]

    def legal_moves(self) -> tuple[Cell, ...]:
        return () if self.winner else tuple((q, r) for r in range(SIZE) for q in range(SIZE)
                                           if (q, r) not in self._board)

    def validate_move(self, player_id: str, q: int, r: int):
        if self.winner:
            raise IllegalMove('The game is already won')
        if player_id != self.current_player:
            raise IllegalMove('Only the current player may act')
        try:
            cell = validate_cell(q, r)
        except ValueError as exc:
            raise IllegalMove(str(exc)) from exc
        if cell in self._board:
            raise IllegalMove('Cell is occupied')

    def play(self, player_id: str, q: int, r: int):
        self.validate_move(player_id, q, r)
        color = self.color(player_id)
        self._board[q, r] = color
        self.turn_index += 1
        path = self.connection_path(color)
        if path:
            self.winner, self.winning_path = player_id, path

    def connection_path(self, color: str) -> tuple[Cell, ...]:
        if color not in COLORS:
            raise ValueError('Unknown color')
        axis = 1 if color == 'red' else 0
        starts = sorted(cell for cell, occupant in self._board.items()
                        if occupant == color and cell[axis] == 0)
        parents: dict[Cell, Cell | None] = {cell: None for cell in starts}
        queue = deque(starts)
        while queue:
            cell = queue.popleft()
            if cell[axis] == SIZE-1:
                path = []
                while cell is not None:
                    path.append(cell)
                    cell = parents[cell]
                return tuple(reversed(path))
            for adjacent in neighbors(*cell):
                if self._board.get(adjacent) == color and adjacent not in parents:
                    parents[adjacent] = cell
                    queue.append(adjacent)
        return ()

    @staticmethod
    def specification() -> dict:
        return {'game': 'Hex', 'size': SIZE, 'turn_order': 'red_first_then_alternating',
                'neighbors_axial': [[1, 0], [-1, 0], [0, 1], [0, -1], [1, -1], [-1, 1]],
                'goals': {'red': 'connect r=0 to r=6', 'blue': 'connect q=0 to q=6'},
                'capture': False, 'pass': False, 'swap_rule': 'disabled_in_v1_fixed_color_ownership',
                'competitive_comparison': 'paired_games_reverse_agent_color_assignments',
                'rules_reference': 'https://www.hexboard.com/Hex_Connections_Rule_Guide_2017.pdf'}
