"""Three board-game levels with black/white ownership and physical commit gates."""
from .geometry import BoardGameGeometry, COLORS, GAME_IDS, geometry_for
from .native_observer import BoardGameNativeObserver
from .protocol import ActuatorOwnership, BoardGamePlayerProtocol, RGBD_MODE, SMOKE_MODE
from .release import prepare_task_release, task_specification, verify_evaluator_sources
from .rules import BoardGameRules, IllegalMove
from .settlement import (BoardGameMatch, NativeFrame, PhysicalSettlementVerifier, PieceEvidence,
                         PlacementIntent, SettlementResult, SettlementThresholds)

__all__ = ['ActuatorOwnership', 'BoardGameGeometry', 'BoardGameMatch', 'BoardGameNativeObserver',
           'BoardGamePlayerProtocol', 'BoardGameRules', 'COLORS', 'GAME_IDS', 'IllegalMove',
           'NativeFrame', 'PhysicalSettlementVerifier', 'PieceEvidence', 'PlacementIntent',
           'RGBD_MODE', 'SMOKE_MODE', 'SettlementResult', 'SettlementThresholds', 'geometry_for',
           'prepare_task_release', 'task_specification', 'verify_evaluator_sources']
