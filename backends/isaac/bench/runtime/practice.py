"""Single-side episode ownership, checked independently by host and native."""
import numpy as np


def policy_targets(session):
    """Original acknowledged commands, not their float32 native readback.

    The classical executor retains its exact submitted reference. Native drive
    storage rounds to float32; replacing that reference at every begin can fail
    its unchanged continuity check even when the native drive bits are equal.
    """
    targets = session._practice_acknowledged_targets
    if targets is None:
        raise RuntimeError('Practice requires initialized acknowledged command targets')
    return dict(targets)


def checked_acknowledgement(session, reply, command):
    """Validate actual native storage before committing a practice command ACK."""
    if getattr(session, 'practice_side', None) is None:
        return None
    targets = command.joint_positions_rad
    actual = reply['observation'].get('robot', {}).get('joint_targets_rad')
    if (not isinstance(actual, dict) or not set(targets) <= set(actual)
            or any(type(actual[n]) not in (int, float) or not np.isfinite(actual[n])
                   or np.float32(actual[n]) != np.float32(value) for n, value in targets.items())):
        raise RuntimeError('Native practice drive differs from the submitted command')
    return dict(targets)


def owned_names(session):
    side = getattr(session, 'practice_side', None)
    if side is None:
        return None
    return {n for n in session.capabilities.joint_limits if n.startswith((side+'_', 'fr3_'+side+'_'))}


def check_geometry(session, value):
    side = getattr(session, 'practice_side', None)
    if side is None:
        return
    from .service import ServiceError
    geometries = value.hand_geometries or {}
    if value.geometry is not None or set(geometries) != {side}:
        raise ServiceError(403, 'Practice geometry may select only the active learner side')
    goal = geometries[side]
    if (not set(goal.hand_posture_rad or {}) <= owned_names(session)
            or goal.fingertip_frame_id not in (None, side+'_hand_flange', 'world')):
        raise ServiceError(403, 'Practice geometry references the inactive robot')


def check_action(session, action):
    side = getattr(session, 'practice_side', None)
    if side is None:
        return
    from bench.protocol.schemas import DirectJointAction
    from .service import ServiceError
    if isinstance(action, DirectJointAction):
        targets = session._acknowledged_drive_targets()
        if targets is None:
            raise ServiceError(503, 'Practice requires complete acknowledged drive targets')
        goal = action.arm_joint_positions_rad | action.hand_joint_positions_rad
        if any(np.float32(value) != np.float32(targets[name]) for name, value in goal.items() if name not in owned_names(session)):
            raise ServiceError(403, 'Practice action changed the inactive robot drive')
    elif action.skill != 'hold':
        check_geometry(session, action)


def check_function(session, name, arguments):
    side = getattr(session, 'practice_side', None)
    if side is None:
        return
    from .service import ServiceError
    if name in ('move_wrist', 'set_hand_targets') and arguments.hand != side:
        raise ServiceError(403, 'Practice function may select only the active learner side')
    if name == 'hand_preset' and arguments.side != side:
        raise ServiceError(403, 'Practice function may select only the active learner side')
    if name in ('move_geometry', 'query_ik', 'query_motion'):
        check_geometry(session, arguments)


def check_receipt(session, score, *, initial=False):
    if getattr(session, 'practice_side', None) is None:
        return
    from bench.protocol.practice import PracticeProgress
    progress = PracticeProgress.model_validate(score['practice_progress'])
    count = len(session.public_task['practice']['goal']['stages'])
    previous = 0 if initial else session.score['practice_progress']['stage_index']
    if (score.get('schema') != 'card_rsi_practice_score.v1' or score.get('formal_game_result') is not False
            or progress.stage_count != count or progress.stage_index != progress.confirmed_stages
            or not previous <= progress.stage_index <= min(count, previous+(0 if initial else 1))
            or score.get('success') != (progress.stage_index == count)):
        raise ValueError('Native practice progress differs from its declared stage contract')
