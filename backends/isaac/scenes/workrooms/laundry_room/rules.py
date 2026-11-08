"""Public laundry goals; numerical values are frozen simulation priors."""
from copy import deepcopy
import hashlib
import json
import math

TASK_IDS = ('towel_flattening', 'towel_folding', 'towel_stacking')
GARMENT_TASK_IDS = ('shirt_stacking', 'trousers_stacking')
GARMENT_FOLDING_TASK_IDS = ('shirt_folding', 'trousers_folding')
CURRENT_TASK_IDS = TASK_IDS + GARMENT_FOLDING_TASK_IDS
ALL_TASK_IDS = TASK_IDS + GARMENT_TASK_IDS + GARMENT_FOLDING_TASK_IDS
TITLES = dict(zip(ALL_TASK_IDS, ('毛巾摊平摆正', '毛巾沿中线对折', '折后毛巾叠放', '已折上衣叠放', '已折长裤叠放', '单件T恤折叠', '单条长裤折叠')))
TASK_SCHEMA = 'laundry-cloth-task-v1'
ASSET_MATERIAL_CONTRACT = 'laundry_asset_material_contract.v1'
LEGACY_PROFILE = 'laundry_strict_task.v1'
PRACTICAL_PROFILE = 'laundry_practical_task.v2'
STACKING_PRACTICAL_PROFILE = 'laundry_practical_stacking_task.v3'
FOLDING_PRACTICAL_PROFILE = 'laundry_practical_folding_task.v3'
GARMENT_LEGACY_PROFILE = 'laundry_folded_garment_task.v1'
GARMENT_PROFILE = 'laundry_folded_garment_task.v2'
GARMENT_PROFILES = (GARMENT_LEGACY_PROFILE, GARMENT_PROFILE)
GARMENT_FOLDING_LEGACY_PROFILE = 'laundry_single_garment_folding_task.v1'
GARMENT_FOLDING_SCALED_PROFILE = 'laundry_single_garment_folding_task.v2'
GARMENT_FOLDING_AIR_PROFILE = 'laundry_single_garment_folding_task.v3'
GARMENT_FOLDING_PROFILE = 'laundry_single_garment_folding_task.v4'
GARMENT_FOLDING_AIR_PROFILES = (GARMENT_FOLDING_AIR_PROFILE, GARMENT_FOLDING_PROFILE)
GARMENT_FOLDING_SCALED_PROFILES = (GARMENT_FOLDING_SCALED_PROFILE, *GARMENT_FOLDING_AIR_PROFILES)
GARMENT_FOLDING_PROFILES = (GARMENT_FOLDING_LEGACY_PROFILE, *GARMENT_FOLDING_SCALED_PROFILES)
TOWEL_PRACTICAL_PROFILES = (PRACTICAL_PROFILE, STACKING_PRACTICAL_PROFILE, FOLDING_PRACTICAL_PROFILE)
PRACTICAL_PROFILES = (*TOWEL_PRACTICAL_PROFILES, *GARMENT_PROFILES, *GARMENT_FOLDING_PROFILES)
GARMENT_THICKNESS_M = {'shirt_stacking': .0016, 'trousers_stacking': .0018}
GARMENT_NOMINAL_MASS_KG = {'shirt_stacking': .139885, 'trousers_stacking': .203887,
                         'shirt_folding': .18, 'trousers_folding': .26}
ENGINEERING_CONTROL = 'engineering_local_grasp'
ROBOT_CONTROL = 'robot_joint_targets'
ROBOT_HAND_EXCURSION_M = .01


def control_mode(task):
    return task.get('evaluation', {}).get('control_mode', ENGINEERING_CONTROL)


def rule_profile(task):
    # Missing profile is the original immutable contract, never a migration.
    return task.get('evaluation', {}).get('rule_profile', LEGACY_PROFILE)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def default_profile(task_id):
    if task_id not in ALL_TASK_IDS:
        raise ValueError('Unknown laundry task')
    return (GARMENT_FOLDING_PROFILE if task_id in GARMENT_FOLDING_TASK_IDS else
            GARMENT_PROFILE if task_id in GARMENT_TASK_IDS else PRACTICAL_PROFILE)


def thresholds(task_id, profile=None):
    profile = default_profile(task_id) if profile is None else profile
    if task_id in GARMENT_FOLDING_TASK_IDS:
        if profile not in GARMENT_FOLDING_PROFILES:
            raise ValueError('Single-garment folding requires its explicit folding profile')
        limits = thresholds('shirt_stacking' if task_id == 'shirt_folding' else 'trousers_stacking', GARMENT_PROFILE)
        limits.update(garment_shell_layer_thickness_ratio_min=.45,
                      garment_shell_layer_fraction_min=.85)
        if profile in GARMENT_FOLDING_SCALED_PROFILES:
            del limits['hard_intersection_pairs']
            limits.update(placement_error_m=.15,
                crossing_pair_length_max_m=.012, crossing_total_length_max_m=.075,
                crossing_component_diameter_max_m=.020, crossing_rest_area_fraction_max=.002,
                crossing_coplanar_area_max_m2=.000001,
                hard_crossing_pair_length_m=.05, hard_crossing_total_length_m=.5,
                hard_crossing_component_diameter_m=.10, hard_crossing_rest_area_fraction=.02,
                hard_crossing_coplanar_area_m2=.00005)
        if profile == GARMENT_FOLDING_PROFILE:
            # Single-item folding allows unpressed fold air and centimetre
            # wrinkles; XY identity, physical thickness and global shape stay
            # independently bounded. Earlier profiles remain immutable.
            limits['garment_layer_gap_error_m'] = .025
        return limits
    if task_id in GARMENT_TASK_IDS:
        if profile not in GARMENT_PROFILES:
            raise ValueError('Garments require their explicit folded-garment rules')
        limits = thresholds('towel_stacking', PRACTICAL_PROFILE)
        limits.update(garment_shape_rms_m=.03, garment_shape_p95_m=.06,
            garment_extent_tolerance_m=.05, garment_height_excess_m=.04,
            garment_projected_area_ratio_min=.80, garment_projected_area_ratio_max=1.20,
            garment_layer_gap_min_m=.0005, garment_layer_gap_ratio_min=.25, garment_layer_gap_error_m=.012,
            garment_layer_xy_error_m=.03, garment_layer_fraction_min=.85,
            garment_table_support_fraction_min=.85, stacking_gap_min_m=.0005)
        if profile == GARMENT_PROFILE:
            # Reset bend radii leave air between layers. Natural settling is
            # bounded by material thickness, not a fraction of that air gap.
            del limits['garment_layer_gap_ratio_min']
            limits['garment_layer_thickness_ratio_min'] = .5
        return limits
    if task_id not in TASK_IDS:
        raise ValueError('Unknown laundry task')
    if profile == STACKING_PRACTICAL_PROFILE and task_id != 'towel_stacking':
        raise ValueError('Practical stacking v3 is exclusive to towel_stacking')
    if profile == FOLDING_PRACTICAL_PROFILE and task_id != 'towel_folding':
        raise ValueError('Practical folding v3 is exclusive to towel_folding')
    if profile not in (LEGACY_PROFILE, *TOWEL_PRACTICAL_PROFILES):
        raise ValueError('Unknown laundry rule profile')
    limits = dict(stable_s=.5, max_time_s=120., max_grasp_patches=2,
        max_grasp_patch_diameter_m=.075,
        max_node_speed_m_s=.035, stable_node_drift_m=.006,
        max_edge_stretch_ratio=1.25, min_triangle_area_ratio=.05,
        max_penetration_m=.001, operation_displacement_m=.025,
        flat_target_rms_m=.018, flat_target_max_m=.045,
        flat_projected_area_ratio_min=.94, flat_height_range_m=.012,
        placement_error_m=.025, folded_pair_rms_m=.018,
        folded_pair_max_m=.035, layer_gap_min_m=.002,
        layer_gap_max_m=.015, layer_order_fraction_min=.95,
        support_gap_max_m=.008, stacking_alignment_m=.025,
        stacking_gap_min_m=.002, stacking_gap_max_m=.012)
    if profile in TOWEL_PRACTICAL_PROFILES:
        limits.update(max_node_speed_m_s=.08, speed_node_percentile=95.,
            stable_node_drift_m=.015, max_penetration_m=.002,
            max_edge_stretch_ratio=1.5, min_triangle_area_ratio=.01,
            flat_target_rms_m=.03, flat_target_max_m=.07,
            flat_projected_area_ratio_min=.90, flat_height_range_m=.025,
            placement_error_m=.05, folded_pair_rms_m=.03, folded_pair_max_m=.06,
            layer_gap_min_m=.001, layer_gap_max_m=.025, layer_order_fraction_min=.85,
            support_gap_max_m=.012, stacking_alignment_m=.05,
            stacking_gap_min_m=.001, stacking_gap_max_m=.020,
            folded_extent_tolerance_m=.05, folded_height_range_m=.04,
            stacking_support_fraction_min=.85, hard_penetration_m=.01,
            hard_edge_stretch_ratio=2., hard_intersection_pairs=8)
    if profile == STACKING_PRACTICAL_PROFILE:
        limits.update(stable_node_drift_m=.008, stability_position_interval_s=.020,
            folded_height_area_percentile=99., folded_height_absolute_max_m=.045)
    if profile == FOLDING_PRACTICAL_PROFILE:
        # A curled free corner need not veto a mostly aligned half-fold.
        # Keep the old 6 cm bound on at least 95% of paired material area,
        # while bounding every outlier and retaining the global 3 cm RMS.
        limits.update(folded_pair_area_percentile=95., folded_pair_percentile_m=.06,
            folded_pair_max_m=.08)
    return limits


def public_rules(task_id, seed=0, profile=None, control_mode=ENGINEERING_CONTROL):
    profile = default_profile(task_id) if profile is None else profile
    if task_id in GARMENT_FOLDING_TASK_IDS:
        return _garment_folding_rules(task_id, seed, profile, control_mode)
    if task_id in GARMENT_TASK_IDS:
        return _garment_rules(task_id, seed, profile, control_mode)
    if task_id not in TASK_IDS or type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Known task and uint32 seed required')
    if (control_mode not in (ENGINEERING_CONTROL, ROBOT_CONTROL)
            or control_mode == ROBOT_CONTROL and profile not in TOWEL_PRACTICAL_PROFILES):
        raise ValueError('Robot towel tasks require practical rules')
    instruction = {
        'towel_flattening': '把起皱、偏置的毛巾摊平，长边沿台面X方向，摆到指定区域并释放。',
        'towel_folding': '沿毛巾长边的中线对折成约30×40厘米，使两半边角基本重合，摆入目标区域后释放。',
        'towel_stacking': '把左侧已折毛巾移到右侧另一条已折毛巾上，两条都保持折形，释放后稳定叠放。',
    }[task_id]
    goals = {
        'towel_flattening': ['同时检查展开面积、整片对应位置、长边朝向、残余高度与台面支撑；允许180度等价朝向。'],
        'towel_folding': ['两半对应位置、两层间距、边界、目标位置和折形同时合格；允许任一半翻折及180度等价朝向。'],
        'towel_stacking': ['两条毛巾均为动态布料且保持对折形态；上层由下层实际布面支撑，下层由折衣台支撑。'],
    }[task_id]
    contract = {'schema': 'physical_task_referee_contract.v1', 'task_id': task_id,
        'instruction_zh': instruction,
        'phases': {'working': '整理任务毛巾。', 'settling': '释放全部抓点并等待稳定。'},
        'reason_descriptions': {'reset_baseline_not_scored': '初态只建立基准，不得分。',
            'task_in_progress': '目标或必要操作尚未完成，可继续调整。',
            'waiting_for_release_and_stability': '需释放全部外部控制并保持稳定。',
            'success': '任务几何、物理证据、操作和释放稳定条件通过。',
            'native_evidence_unknown': '原生记录缺失或不连续，本次不评分。',
            'physics_safety_stop': '穿透、自交、异常拉伸或其他物理异常。',
            'incomplete_budget': '预算耗尽，任务未完成。'},
        'success_conditions': goals+['重力和碰撞保持启用，无隐藏固定点；所有抓点释放后连续稳定0.5秒。',
            '初态不能直接得分；必须有明确操作及至少2.5厘米的毛巾位移。'],
        'recovery': ['允许重抓、纠偏与重新整理；未达到目标时可继续。'],
        'thresholds': thresholds(task_id, profile),
        'outcomes': {'ongoing': '可以继续。', 'success': '本次控制模式下完成。',
            'task_failure': '任务失败。', 'evaluator_invalid': '证据不完整，不评分。',
            'safety_stop': '物理异常停止。', 'budget_exhaustion': '未完成。', 'cancelled': '取消。'},
        'physical_actions': '本轮工程验证允许最多两个世界空间直径不超过75毫米的局部抓区；一个抓区可同时夹上下层，每个相连材料片在静止网格中的直径也不超过75毫米，每条布控制点不超过10%。验收前必须全部释放。直接驱动布点不代表机器人抓取成功；普通机器人动作接口不因此接受顶点写入。',
        'observation_scope': '规则和标称物性公开；完整布节点、控制掩码、层次和物理审计为私有证据。没有布料接触力或张力测量。',
        'object_properties': [{'id': name, 'label': '动态厚毛巾', 'nominal_mass_kg': .108,
            'basis': '60×40厘米、450克/平方米、厚3毫米均为仿真先验，未经实物标定。'}
            for name in (('towel', 'towel_base') if task_id == 'towel_stacking' else ('towel',))]}
    if profile in TOWEL_PRACTICAL_PROFILES:
        contract['reason_descriptions'].update(
            success='形状、位置与支撑合格，全部释放后保持稳定；小幅仿真抖动不单独判失败。',
            contact_recovery_required='局部交叉、接触或变形暂未恢复，可继续整理；恢复后重新计稳定时间。',
            physics_safety_stop='明显穿透、异常大幅拉伸或大范围交叉，停止本次任务。')
        contract['success_conditions'] += [
            '允许轻微褶皱和边角不齐：摊平/对折对应误差RMS不超过3厘米；对折与叠放中心误差不超过5厘米。',
            '稳定速度按每条毛巾、每一帧的节点速度P95判断（不超过8厘米/秒），同时检查相邻步位置差；所有节点在0.5秒内漂移不超过1.5厘米。',
            '允许最多2毫米接触包络压入、1.5倍局部边长；最终稳定期间无实际布面交叉，叠放几何支撑比例至少85%。',
            '严格物理诊断独立记录，不以单个抖动节点或整理中已恢复的小接触异常一票否决。']
        contract['recovery'] += ['局部接触异常、放置不准或释放后滑动可调整后重试；稳定时间从恢复并完全松手后重新计算。']
    result = {'schema': 'laundry_rules.v1', 'task_id': task_id, 'instruction_zh': instruction,
        'referee': contract, 'reset_credit': False, 'episode_seed': seed,
        'randomization': 'Frozen canonical reset; seed is provenance only.',
        'qualification_scope': 'Engineering cloth manipulation; robot grasping and material calibration are separate.'}
    if profile in TOWEL_PRACTICAL_PROFILES:
        result.update(schema='laundry_rules.v2', rule_profile=profile,
            tolerance_basis='User-authorized practical simulation tolerances; engineering priors, not material calibration.')
    if profile == STACKING_PRACTICAL_PROFILE:
        result.update(schema='laundry_rules.v5', rule_profile=profile)
        conditions = contract['success_conditions']
        conditions[conditions.index('稳定速度按每条毛巾、每一帧的节点速度P95判断（不超过8厘米/秒），同时检查相邻步位置差；所有节点在0.5秒内漂移不超过1.5厘米。')] = (
            '稳定速度按每条毛巾每一原生帧与20毫秒前真实位置差计算，节点速度P95不超过8厘米/秒；全部节点在连续0.5秒窗口相对窗口起点漂移不超过8毫米。须有连续原生帧及额外20毫秒前史；原始速度和相邻步位置速度完整保留诊断。')
        conditions += ['每条折后毛巾按静止三角面积均分到顶点的权重计算高度P99，减全节点最低高度不超过40毫米；全节点绝对高度差仍不得超过45毫米。仅允许不超过1%材料面积的边缘轻卷，其余对应关系、层序、支撑、自交和释放要求不变。']
    if profile == FOLDING_PRACTICAL_PROFILE:
        result.update(schema='laundry_rules.v6', rule_profile=profile)
        contract['success_conditions'] += ['对折两半XY对应误差按每对两端的静止材料面积权重计算：RMS不超过3厘米、P95不超过6厘米、全点绝对最大误差不超过8厘米。允许少量自由边角卷起，但轮廓、高度、统一层序、支撑、物理与完整释放稳定要求不变；不得用平均值掩盖漏折或大范围错位。']
    if control_mode == ROBOT_CONTROL:
        result.update(control_mode=control_mode,
            qualification_scope='Robot joint actions with native geometry evidence; grasp forces and material calibration are not measured.')
        contract['physical_actions'] = '一个Agent同时控制两台FR3和官方左、右SHARPA，共58个关节。只接受机器人关节目标；不允许直接写布节点、隐藏固定点或工程抓持。允许重抓与整理，不要求固定的双手动作顺序。'
        contract['phases']['settling'] = '双手和机器人全部离开毛巾，等待放置稳定。'
        contract['success_conditions'] += [
            '任务毛巾连续处于实际手部碰撞形状的保守几何接近带期间，至少一只手相对该段起点移动1厘米，且任务毛巾在同一连续接近段内有至少2.5厘米位移；静止接近、空中挥手或纯被动下落不得分。',
            '释放以全部机器人碰撞形状与全部毛巾的几何分离为准；不要求零抖动，不测量夹持力。']
    return result


def _garment_rules(task_id, seed, profile, mode):
    """New public identity; the three historical towel dictionaries stay exact."""
    limits = thresholds(task_id, profile)
    label = '上衣' if task_id == 'shirt_stacking' else '长裤'
    template = public_rules('towel_stacking', seed, PRACTICAL_PROFILE, mode)
    # The native cloth IDs deliberately retain their existing observer meaning.
    def relabel(value):
        if isinstance(value, str):
            return value.replace('毛巾', label)
        if isinstance(value, list):
            return [relabel(v) for v in value]
        if isinstance(value, dict):
            return {k: relabel(v) for k, v in value.items()}
        return value
    result = relabel(template)
    instruction = f'把左侧已折{label}移到右侧另一件已折{label}上，两件都保持原折叠形态，释放后稳定叠放。'
    result.update(schema='laundry_rules.v3', task_id=task_id, rule_profile=profile,
        instruction_zh=instruction)
    referee = result['referee']
    referee.update(task_id=task_id, instruction_zh=instruction, thresholds=limits,
        object_properties=[{'id': name, 'label': '动态已折'+label,
            'nominal_mass_kg': GARMENT_NOMINAL_MASS_KG[task_id],
            'basis': f'真实衣片轮廓的原生柔性表面；标称厚度{GARMENT_THICKNESS_M[task_id]*1000:g}毫米，质量由冻结衣片面积和面密度先验估算。厚度、质量、材料与折叠参考均未经实物标定。'}
            for name in ('towel', 'towel_base')])
    referee['success_conditions'] = [
        f'两件{label}均为动态布料；保持各自冻结参考的折叠形态，不得展开、严重拉伸、塌陷或穿插。',
        '允许整件在台面内旋转及轻微褶皱：刚体对齐后的形状RMS不超过3厘米，位置中心误差不超过5厘米；不要求每个边角完美重合。',
        '检查参考对应材料层的相对位置和层距，至少85%保有合理层次且层距不小于参考的25%；上件至少85%的底面采样由下件实际布面支撑，下件至少85%的底面采样由台面支撑。',
        '重力和碰撞保持启用，无隐藏固定点；全部外部控制释放后连续稳定0.5秒。',
        '稳定按每件、每帧原始和相邻步位移推算速度P95不超过8厘米/秒判断；所有节点在0.5秒内漂移不超过1.5厘米。',
        '允许最多2毫米接触包络压入和1.5倍局部边长；稳定期间无实际布面交叉。',
        '初态不能直接得分；需明确操作及任务衣物至少2.5厘米位移。']
    if profile == GARMENT_PROFILE:
        referee['success_conditions'][2] = (
            '检查参考对应材料层的相对位置和正向层序，至少85%保有合理层次；层距下限为0.5毫米与标称材料厚度一半的较大值，允许初始空气层压实。'
            '上件至少85%的底面采样由下件实际布面支撑，下件至少85%的底面采样由台面支撑。')
    if mode == ROBOT_CONTROL:
        referee['success_conditions'] += [
            '至少一只手在同一连续实际碰撞几何接近段内净移动1厘米，且任务衣物在该段内位移至少2.5厘米；被动下落、空中挥手或静止接近均不计操作。',
            '释放要求全部机器人碰撞形状与两件衣物完全几何分离，随后重新计算0.5秒稳定时间。']
    return result


def _garment_folding_rules(task_id, seed, profile, mode):
    """Single unfolded item, folded through actual material motion then released."""
    limits = thresholds(task_id, profile)
    label = 'T恤' if task_id == 'shirt_folding' else '长裤'
    # Keep the mature release/recovery/control contract; replace the operation
    # and every two-garment stacking condition with the single-item goal.
    result = deepcopy(public_rules('towel_folding', seed, PRACTICAL_PROFILE, mode))
    def relabel(value):
        if isinstance(value, str):
            return value.replace('毛巾', label)
        if isinstance(value, list):
            return [relabel(v) for v in value]
        if isinstance(value, dict):
            return {k: relabel(v) for k, v in value.items()}
        return value
    result = relabel(result)
    fold = ('将左右袖部向内收拢，再将下摆向领口方向对折' if task_id == 'shirt_folding'
            else '先将两条裤腿对齐折在一起，再将裤脚向腰部方向对折')
    instruction = f'把台面上展开的单件{label}整理折好：{fold}，形成紧凑、基本对齐的折形，放在台面目标区域并完全松手。'
    result.update(schema='laundry_rules.v4', task_id=task_id, rule_profile=profile,
                  instruction_zh=instruction)
    contract = result['referee']
    contract.update(task_id=task_id, instruction_zh=instruction, thresholds=limits,
        object_properties=[{'id': 'towel', 'label': '动态展开单件'+label,
            'nominal_mass_kg': GARMENT_NOMINAL_MASS_KG[task_id],
            'basis': '单件缝合衣物的原生柔性表面；材料厚度限0.5至2毫米，具体厚度、质量、网格和材料先验以冻结资产为准，未作实物标定。'}])
    contract['success_conditions'] = [
        f'初态只有一件展开的{label}；必须实际折叠本件衣物，不是搬运已折成品，也不是叠到另一件衣物上。',
        '不限定抓取点和动作先后，只检查最终折叠结果；衣物可在台面内整体旋转和小幅偏移，允许自然褶皱与边角不完全重合。',
        '按材料对应关系检查整件折形：刚体对齐后的形状RMS不超过3厘米、形状误差P95不超过6厘米、目标中心误差不超过5厘米；不能只压成小团或只缩小外包围框。',
        '至少85%的对应材料层具有正确层次和相近的平面位置；层距至少为0.5毫米和名义厚度一半中的较大值。至少85%的目标底面采样由原有台面实际支撑。',
        '缝合衣物的天然前后衣片另行检查：至少85%的冻结对应点保留不小于材料厚度45%的有向层距，不能把天然双面完全压成零厚度；长裤允许任一裤腿翻到另一裤腿上。',
        '重力和碰撞保持启用，无隐藏固定点；完全释放后保持连续0.5秒，原始/位置差分节点速度P95均不超过8厘米/秒，所有节点漂移不超过1.5厘米。',
        '允许最多2毫米接触包络压入和1.5倍局部边长；最终稳定窗口不得有实际布面交叉、持续穿透、严重拉伸或塌陷。',
        '初态、被动下落和无衣物动作不能得分；需实际控制操作与至少2.5厘米材料位移。']
    if mode == ROBOT_CONTROL:
        contract['success_conditions'] += [
            '至少一只手在同一连续实际手部几何接近段内净移动1厘米，且衣物在该段内移动至少2.5厘米；接触离开后不能追认之前的被动位移。',
            '全部机器人碰撞形状与本件衣物分离后才开始释放稳定计时；不要求零抖动或固定双手动作顺序。']
    if profile in GARMENT_FOLDING_SCALED_PROFILES:
        contract['success_conditions'][2] = '按材料对应关系检查整件折形：刚体对齐后的形状RMS不超过3厘米、形状误差P95不超过6厘米；允许在台面参考区域中心15厘米内放置，仍须完整折形与实际台面支撑，不能只压成小团或只缩小外包围框。'
        contract['success_conditions'][6] = '允许最多2毫米接触包络压入和1.5倍局部边长。最终可保留少量局部布面交叉，但须同时满足：单对交线不超过12毫米、逐对保守总长不超过75毫米、连通缺陷簇直径不超过20毫米、涉及的唯一材料三角形面积不超过整件的0.2%、共面重叠面积之和不超过1平方毫米；原始交叉对数完整保留，不凭对数放行。'
        contract['success_conditions'].append('较大的局部交叉须恢复后重新计稳定；交线单对超过5厘米、保守总长超过50厘米、单簇直径超过10厘米、受影响材料面积超过2%或共面重叠面积之和超过50平方毫米会停止任务。缺失或截断的交叉尺度证据不得分。')
        contract['reason_descriptions']['contact_recovery_required'] = '交叉尺度、接触或局部变形尚超出实用容差，可继续整理；恢复后重新计稳定时间。'
    if profile in GARMENT_FOLDING_AIR_PROFILES:
        contract['success_conditions'][3] = '至少85%的对应材料层具有正确上下顺序和平面位置；层距至少为0.5毫米和名义厚度一半中的较大值，最多为参考层距加12毫米。允许参考折弧内的空气层自然压实，无须保持参考空气隙；至少85%的目标底面采样须由原有台面实际支撑。'
    if profile == GARMENT_FOLDING_PROFILE:
        contract['success_conditions'][3] = '至少85%的对应材料层具有正确上下顺序，XY相对位置误差不超过3厘米；层距至少为0.5毫米和名义厚度一半中的较大值，最多为参考层距加25毫米。单件折叠允许自然褶皱和未压平的折层空气，也允许空气层压实；整件形状、额外高度不超过4厘米和天然衣片厚度仍须分别合格。至少85%的目标底面采样须由原有台面实际支撑。'
    return result


def configuration_rules(task_id, seed, profile, configuration_id, *, asset_metadata=None):
    """Explicit operator/material declarations; old calls remain byte-exact."""
    if task_id not in CURRENT_TASK_IDS or configuration_id not in ('single_arm', 'dual_same_side', 'dual_opposed'):
        raise ValueError('Known current laundry task and installation required')
    result = public_rules(task_id, seed, profile, ROBOT_CONTROL)
    result['robot_configuration_id'] = configuration_id
    if configuration_id == 'single_arm':
        contract = result['referee']
        contract['physical_actions'] = '一个Agent控制一台FR3和官方右SHARPA，共29个关节。只接受机器人关节目标；不允许直接写布节点、隐藏固定点或工程抓持。允许单手重抓、借助台面整理，不要求双手同时操作。'
        contract['phases']['settling'] = '右手和全部机器人碰撞形状离开所有任务衣物，等待放置稳定。'
    if asset_metadata is not None:
        if asset_metadata.get('task_id') != task_id:
            raise ValueError('Public material declaration requires the selected asset task')
        contract = result['referee']
        labels = {row['id']: row['label'] for row in contract['object_properties']}
        cloths = asset_metadata['cloths']
        if {row['id'] for row in cloths} != set(labels) or len(cloths) != len(labels):
            raise ValueError('Public material inventory differs from the selected cloths')
        properties, materials = [], []
        for row in cloths:
            mass, thickness = row['mass_kg'], row['physical_thickness_m']
            if not all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in (mass, thickness)):
                raise ValueError('Public material mass and thickness must be positive finite priors')
            properties.append({'id': row['id'], 'label': labels[row['id']], 'nominal_mass_kg': mass,
                'basis': f'冻结资产的名义厚度{thickness*1000:g}毫米、质量{mass*1000:g}克；材料与求解器参数为仿真先验，未经实物标定。'})
            materials.append({'id': row['id'], 'physical_thickness_m': thickness, 'mass_kg': mass})
        contract['object_properties'] = properties
        contract['phases']['settling'] = '机器人与全部任务衣物完全分离，等待放置稳定。'
        result['material_contract'] = {'schema': ASSET_MATERIAL_CONTRACT,
            'material_profile': asset_metadata.get('material_profile'), 'cloths': materials}
        if task_id == 'towel_flattening' and asset_metadata.get('reset_profile') == 'towel_partial_edge_fold_v1':
            instruction = '把局部折回的毛巾展开、摊平，长边沿台面X方向，摆到指定区域并完全释放。'
            result['instruction_zh'] = contract['instruction_zh'] = instruction
    return result


def task_public_rules(task):
    material_contract = task.get('public_material_contract')
    if material_contract not in (None, ASSET_MATERIAL_CONTRACT):
        raise ValueError('Unknown public asset material contract')
    if 'robot_configuration_id' in task:
        if control_mode(task) != ROBOT_CONTROL:
            raise ValueError('A configured installation requires ordinary robot joint control')
        return configuration_rules(task['id'], task['episode_seed'], rule_profile(task), task['robot_configuration_id'],
            asset_metadata=task['asset_metadata'] if material_contract is not None else None)
    if material_contract is not None:
        raise ValueError('Public asset material declarations require an explicit configured release')
    return public_rules(task['id'], task['episode_seed'], rule_profile(task), control_mode(task))


def bind_task_rules(task):
    task = deepcopy(task)
    task['public_rules_sha256'] = digest(task['public_rules'])
    task['public_referee_contract_sha256'] = digest(task['public_rules']['referee'])
    task['public_rules_file'] = 'rules.json'
    return task
