"""Finite reset-only curricula. These specify goals, never robot trajectories."""
from copy import deepcopy
from dataclasses import asdict
import math

from bench.protocol.practice import PracticeSelection, PublicPracticeCase
from bench.protocol.task_referee import PublicContract
from scenes.board_games.expansion_release import canonical_hash
from ..expansion_referee import ExpansionThresholds, public_rule_contract
from ..expansion_tasks import RELEASES
from ..game_designs import TASK_IDS, design_for
from ..expansion_rules import rotated_cells

GYM_IDS = tuple(name+'_practice' for name in TASK_IDS)
STRATA = {
    'polyomino_territory': ('mono_translate', 'domino_rotate', 'concave_rotate', 'multi_piece_adjacent'),
    'barrier_race': ('pawn_step', 'wall_place', 'wall_rotate', 'pawn_obstacle'),
    'memory_pairs': ('flip', 'restore', 'collect_pair', 'flip_restore_collect'),
    'disk_curling': ('launch', 'release_line', 'target_score', 'obstacle_contact'),
}
VARIANTS = {'train': range(4), 'validation': range(4, 6)}
REASONS = {
    'awaiting_native_evidence': '等待完整原生物理采样。',
    'baseline': '初态已观测；尚未完成任何练习。',
    'own_hand_operation_required': '指定对象需要本侧手部真实接触及足够运动。',
    'target_required': '当前阶段的公开目标尚未满足。',
    'release_support_required': '需要真实支撑、机器人释放及稳定。',
    'inventory_recovery_required': '非当前操作对象需要恢复其已确认位置并稳定。',
    'settling': '目标已达到，正在累计连续释放稳定时间。',
    'stage_confirmed': '一个公开目标阶段由原生物理确认。',
    'success': '所有练习目标阶段已物理确认；不代表正式游戏胜负。',
    'missing_physical_evidence': '原生证据不完整，不能认定成功。',
    'physical_failure': '原生穿透或物理证据异常。',
    'rule_violation': '违反单侧操作或公开推盘规则。',
    'interrupted': '练习中断，未完成的目标不授予成功。',
    'preparing_disk': '等待符合正式推盘条件的手部接触推进。',
    'free_crossing_required': '需要在线前释放并自由越线。',
    'obstacle_contact_required': '需要圆盘之间的真实原生接触。',
    'score_required': '本次静止圆盘未达到公开最低得分。',
}


def parent_id(gym_id):
    if gym_id not in GYM_IDS:
        raise ValueError('Unknown card practice task')
    return gym_id.removesuffix('_practice')


def selections(gym_id, *, split=None):
    task_id = parent_id(gym_id)
    return [dict(active_side=side, case_id=f'{stratum}_{i:02d}', split=part)
            for part in VARIANTS if split is None or part == split
            for side in ('left', 'right') for stratum in STRATA[task_id] for i in VARIANTS[part]]


def rules_for(gym_id):
    task_id = parent_id(gym_id)
    instruction = ('单侧自主练习：仅操作本次 active_side 的原机器人，按公开目标依次完成真实操作；'
        '另一侧保持原生已确认驱动。目标不是抓取轨迹，机器人关节动作由 Agent 自行决定。')
    contract = {
        'schema': 'physical_task_referee_contract.v1', 'task_id': gym_id,
        'instruction_zh': instruction,
        'phases': {'practice': '完成当前公开阶段；所有阶段确认后结束。', 'complete': '练习成功。', 'incomplete': '练习未完成。'},
        'reason_descriptions': REASONS,
        'success_conditions': ['全部原生子步完整，保持原机器人、棋具、材料和数值阈值。',
            '位移/转动型操作须在一段连续有效本侧手接触内达到原 5 mm 净位移或 10° 净转角；从首次接触子步前一帧计，不能追认段前后无手运动，也不累加往返路径长度。未达标即脱离可重新抓取开启新段；已取得操作信用可保留到松手稳定。推盘另按原有效推进速度/单次接触/释放越线判据，不追加此位移转角门槛。不能靠初态或 RPC 目标授予操作。',
            '指定目标的真实几何满足条件，真实 Deck 或允许的收藏支撑链、释放及连续 300 ms 稳定。',
            '其他库存恢复到初态或先前已确认位置；圆盘障碍可被圆盘碰撞推动。'],
        'recovery': ['未达到目标可继续用普通机器人动作调整。', '预算耗尽或主动结束只保留未完成结果，不虚构成功。'],
        'thresholds': asdict(ExpansionThresholds()),
        'outcomes': {key: text for key, text in (
            ('ongoing', '练习进行中'), ('success', '仅本次练习目标成功'), ('task_failure', '练习规则违反'),
            ('evaluator_invalid', '原生证据无效'), ('safety_stop', '物理异常'),
            ('budget_exhaustion', '预算中断，目标未完成'), ('cancelled', '取消，目标未完成'))},
        'physical_actions': '仅普通公开机器人关节/几何动作；禁止物体状态写入、外力、隐藏附着和人给抓取轨迹。',
        'observation_scope': 'RGB-D、机器人 proprio 和粗粒度合法反馈；不提供实时棋具位姿、速度、载荷、碰撞真值或隐藏配对。',
    }
    contract = PublicContract.model_validate(contract).model_dump(mode='json', by_alias=True)
    if task_id == 'disk_curling':
        contract['success_conditions'].extend([
            '准备期间整盘留在本侧发射区；有效推进必须为盘平放、Deck 支撑、本侧手接触且沿目标方向速度达到公开阈值。',
            '一次推进只允许连续一次手接触，在盘体中心到发射线前释放；释放后再接触或手臂推进违规。',
            'launch 练习只要求有效推进后释放停稳；其余 shoot 阶段按 require_line_crossing 要求释放后自由越线。',
            '按最低分和障碍接触目标验收；正分只来自 Deck 上真实支撑停稳的圆盘，非计分表面必须真实支持且只能零分。'])
    return {'referee': contract, 'parent_game_rules': public_rule_contract(task_id),
        'curriculum': {'strata': list(STRATA[task_id]), 'active_sides': ['left', 'right'],
            'train_variants': list(VARIANTS['train']), 'validation_variants': list(VARIANTS['validation']),
            'reset_only': True, 'no_prescribed_robot_trajectory': True,
            'memory': 'Only cards actually face up in the reset may be selected as a known matching pair. Other pair assignments are private.'}}


def _quaternion(degrees):
    angle = math.radians(degrees)/2
    return [math.cos(angle), 0., 0., math.sin(angle)]


def _target(name, position=None, quaternion=None, **kwargs):
    return {'object_id': name, 'position_m': position, 'quaternion_wxyz': quaternion, **kwargs}


def build_case(gym_id, selection, metadata):
    """Return host-only case; only ``public`` may cross the Agent boundary."""
    task_id = parent_id(gym_id)
    selection = PracticeSelection.model_validate(selection).model_dump()
    side, split, case_id = (selection[k] for k in ('active_side', 'split', 'case_id'))
    if selection not in selections(gym_id):
        raise ValueError('Practice selection is absent from the finite frozen curriculum')
    stratum, variant_text = case_id.rsplit('_', 1)
    variant = int(variant_text)
    color, sign = ('black', -1) if side == 'left' else ('white', 1)
    design = design_for(task_id)
    if metadata.get('task_id') != task_id or metadata.get('dimensions') != design:
        raise ValueError('Practice requires the original scene-v2 asset metadata')
    specs = {p['id']: p for p in design['pieces']}
    reset = {name: {'position_m': list(metadata['initial_positions_m'][name]),
                    'quaternion_wxyz': list(metadata['initial_orientations_wxyz'][name])}
             for name in specs}
    stages = []
    if task_id == 'polyomino_territory':
        shape = {'mono_translate': 'mono', 'domino_rotate': 'domino',
                 'concave_rotate': 'tet_L', 'multi_piece_adjacent': 'tet_L'}[stratum]
        names = [color+'_'+shape]
        rotation = 0 if shape == 'mono' else 90 if variant % 2 == 0 else 270
        if stratum == 'multi_piece_adjacent':
            names.append(color+'_mono')
            rotation = 90
        cells = rotated_cells(specs[names[0]]['cells'], rotation)
        width, height = max(c[0] for c in cells)+1, max(c[1] for c in cells)+1
        anchor = [variant % (7-width), 0 if side == 'left' else 6-height]
        cells = [[c[0]+anchor[0], c[1]+anchor[1]] for c in cells]
        targets = []
        for i, name in enumerate(names):
            selected = cells if i == 0 else [[anchor[0]+1, anchor[1]+1]]
            center = [.65+((min(c[0] for c in selected)+max(c[0] for c in selected))/2-2.5)*.052,
                      ((min(c[1] for c in selected)+max(c[1] for c in selected))/2-2.5)*.052, .817]
            targets.append(_target(name, center, _quaternion(rotation if i == 0 else 0), cells=selected))
            reset[name]['position_m'][0] += (variant-2.5)*.004
        # Adjacent placements are separate opportunities for fresh hand contact.
        stages = [{'operation': 'place', 'targets': [target]} for target in targets]
    elif task_id == 'barrier_race':
        if stratum in ('pawn_step', 'pawn_obstacle'):
            name = color+'_pawn'
            q = 1+variant % 3
            r = 0 if side == 'left' else 4
            reset[name]['position_m'] = [.65+(q-2)*.075, (r-2)*.075, .8491]
            reset[name]['quaternion_wxyz'] = _quaternion(variant*10)
            destination = [q, r+(1 if side == 'left' else -1)]
            if stratum == 'pawn_obstacle':
                # A real passive wall beside the destination; no prescribed route.
                wall = ('white' if color == 'black' else 'black')+'_barrier_0'
                reset[wall]['position_m'] = [.65+(q-1.5)*.075, (r+(0.5 if side == 'left' else -.5)-2)*.075, .8341]
                reset[wall]['quaternion_wxyz'] = _quaternion(90)
            stages = [{'operation': 'place', 'targets': [_target(name,
                [.65+(destination[0]-2)*.075, (destination[1]-2)*.075, .8485], orientation='upright')]}]
        else:
            name = color+'_barrier_0'
            q, r = variant % 4, 0 if side == 'left' else 3
            rotation = 90 if stratum == 'wall_rotate' else 0
            reset[name]['position_m'][0] += (variant-2.5)*.004
            stages = [{'operation': 'place', 'targets': [_target(name,
                [.65+(q-1.5)*.075, (r-1.5)*.075, .8335], _quaternion(rotation))]}]
    elif task_id == 'memory_pairs':
        if stratum in ('flip', 'restore'):
            name = f'card_{(variant+(0 if side == "left" else 6)) % 12:02d}'
            if stratum == 'restore':
                reset[name]['quaternion_wxyz'] = [0., 1., 0., 0.]
            stages = [{'operation': stratum, 'targets': [_target(name,
                [*reset[name]['position_m'][:2], .816], orientation='face_up' if stratum == 'flip' else 'face_down')]}]
        else:
            pair = f'pair_{variant}'
            names = sorted(name for name, value in metadata['private_reset_pairs'].items() if value == pair)
            if len(names) != 2:
                raise ValueError('Authored face patterns must contain exactly two matching cards')
            for name in names:
                reset[name]['quaternion_wxyz'] = [0., 1., 0., 0.]
            zone = design['collection_zones'][color]
            bounds = [[zone['center_xy_m'][i]+s*zone['size_xy_m'][i]/2 for i in (0, 1)] for s in (-1, 1)]
            if stratum == 'flip_restore_collect':
                for operation, orientation in (('restore', 'face_down'), ('flip', 'face_up')):
                    stages.append({'operation': operation, 'targets': [_target(name,
                        [*reset[name]['position_m'][:2], .816], orientation=orientation) for name in names]})
            stages.append({'operation': 'collect', 'targets': [_target(name,
                orientation='face_up', region_bounds_xy_m=bounds) for name in names]})
    else:
        name = color+'_disk_'+str(STRATA[task_id].index(stratum))
        # Keep the complete inventory; only reset the selected disk inside its
        # true launch region, away from its three original supply neighbours.
        reset[name]['position_m'][0] += (variant-2.5)*.004
        obstacles = []
        if stratum == 'obstacle_contact':
            obstacle = ('white' if color == 'black' else 'black')+'_disk_0'
            reset[obstacle]['position_m'] = [reset[name]['position_m'][0], -sign*.06, .8156]
            obstacles = [obstacle]
        stages = [{'operation': 'shoot', 'targets': [_target(name, orientation='flat')],
            'minimum_points': 1 if stratum == 'target_score' else 0,
            'obstacle_ids': obstacles, 'require_obstacle_contact': bool(obstacles),
            'require_line_crossing': stratum != 'launch'}]
    goal = {'frame_id': 'world_m_z_up', 'stages': stages,
        'description': '依次完成公开目标；目标坐标是静态任务要求，不是实时棋具真值。每阶段必须本侧手部真实操作并释放稳定。'}
    public = {**selection, 'schema': 'card_rsi_practice_case.v1', 'gym_id': gym_id,
        'parent_task_id': task_id, 'stratum': stratum, 'goal': goal,
        'apparatus': {'board_top_z_m': design['board']['top_z_m'],
            'deck_bounds_xy_m': design['board']['deck_bounds_xy_m'],
            'pieces': [{'object_id': n, 'kind': spec['kind'], 'dimensions_m': spec['dimensions_m'],
                'mass_kg': metadata['masses_kg'][n], 'collision_boxes': spec.get('collision_boxes', [])} for n, spec in specs.items()],
            **({'launch_region_bounds_xy_m': design['board']['launch_region_bounds_xy_m'][color],
                'launch_line_y_m': design['board']['launch_lines_y_m'][color], 'launch_direction_y': -sign,
                'scoring_center_xy_m': design['board']['target_center_xy_m'],
                'scoring_radii_m': design['board']['scoring_radii_m'], 'scoring_points': design['board']['scoring_points']} if task_id == 'disk_curling' else {})},
        'parent_release_sha256': RELEASES[task_id]['manifest_sha256'],
        'public_rules_sha256': canonical_hash(rules_for(gym_id)), 'case_sha256': '0'*64}
    public = PublicPracticeCase.model_validate(public).model_dump(mode='json', by_alias=True)
    public['case_sha256'] = canonical_hash({k: v for k, v in public.items() if k != 'case_sha256'})
    # Never publish either private hash: memory maps have low entropy and a
    # public commitment would otherwise be an offline identity oracle.
    content = {'parent_release_sha256': public['parent_release_sha256'], 'reset': reset,
        'active_side': side, 'goal': public['goal'], 'private_pairs': metadata.get('private_reset_pairs')}
    case = {'schema': 'card_rsi_private_case.v1', 'public': public, 'reset': reset,
        'case_fingerprint': canonical_hash(content)}
    case['private_case_sha256'] = canonical_hash(case)
    return case


def verify_case(case, metadata):
    public = case['public']
    selection = {key: public[key] for key in PracticeSelection.model_fields}
    if case != build_case(public['gym_id'], selection, metadata):
        raise ValueError('Practice case differs from its finite frozen reset/goal declaration')
    return deepcopy(case)
