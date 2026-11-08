"""Pure, source-pinned adaptation of frozen single-agent scorers to two hands.

Only the derivative publisher calls this module. The caller must first verify
the complete base release, then freeze the returned sources, rule hashes and
native-observer configuration together. Keys in ``evaluator_sources`` are paths
relative to the release's ``evaluator/`` directory, without that prefix.
Historical releases and the project family evaluators are never rewritten.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import PurePosixPath


TASK_IDS = {
    'mechanical_assembly': ('parts_sorting', 'peg_insertion', 'shaft_spacer_assembly',
        'fastener_assembly', 'gear_train_assembly', 'nist_connector_insertion',
        'nist_belt_drive_assembly', 'nist_cable_routing', 'nist_wire_harness'),
    'laboratory': ('sample_sorting', 'vial_racking', 'cap_placement',
        'solid_weighing', 'pipette_transfer', 'tube_clamp_transfer'),
    'study': ('stationery_sorting', 'book_stacking', 'book_retrieval',
        'book_shelving', 'drawer_retrieval', 'reading_stand'),
    'living_room': ('remote_operation', 'water_pouring', 'tower_building', 'storage_box'),
    'card_room': ('task0',),
}
ALLOWED_TASKS = frozenset((family, task_id) for family, ids in TASK_IDS.items() for task_id in ids)
HAND_PREFIXES = ['left_', 'right_']
FINGER_PREFIXES = [side+'_'+finger+'_' for side in ('left', 'right')
                   for finger in ('thumb', 'index', 'middle', 'ring', 'pinky')]

# Only the exact reviewed frozen versions may be rewritten. Other source files
# are passed through verbatim, under the caller's complete base-manifest check.
_MECHANICAL_RULES_SHA = 'b1eb92d849db89ba126275036aee347e851c2e84daa2e4057bf080f76d5205a4'
SOURCE_PINS = {
    **{('mechanical_assembly', name): {'rules.py': _MECHANICAL_RULES_SHA}
       for name in ('parts_sorting', 'peg_insertion', 'shaft_spacer_assembly')},
    ('mechanical_assembly', 'fastener_assembly'): {
        'sequence_evaluator.py': '3db8ca2cddca0dc5597d3733c8d4510e4b064bc9a9005994d978053dd56cd28e'},
    ('mechanical_assembly', 'gear_train_assembly'): {
        'sequence_evaluator.py': 'ae00707b2f27508239dbd456de4cf258adf1f71ded63943a6c985a6eb3a0618c'},
    ('card_room', 'task0'): {
        'rules.py': '20add0c40cfd12dd30fbc57865c92b583424099932c4040a83916ff7634356e7'},
    ('living_room', 'water_pouring'): {
        # Original v5 and v6's particle-rest-height spill classification. Both
        # derivatives only replace the single-arm wording; numeric rules and
        # the selected base's physical object contract remain unchanged.
        'rules.py': ('4e3cc275bf8d55bd01b1317b2ee672dc7939efafab1f1dc41b82c51ef96a41f9',
                     '68337a1a9ecfe52f464dc0212b10431ad1b36f2e8d6112e97aea58b35c56020c')},
    ('living_room', 'remote_operation'): {
        'rules.py': ('b3684885a4ad5f1ac3cab603ac39b32f75eaf7b23ecbba04d54b8ac0ab16ac82', '8a90193367056785eb8c421f263d019f66d88191f39d765225023ab3dc2b88ae', '0d3acb43c7ac0dded40a1e0445d26b09f560c89079ce083702bd2168032d72b3'),
        'sequence_evaluator.py': ('c19984711ceca3fe9e81459d8621c5f89d86114e7aa8ac3fd5457b023d403ac5', 'df9053a9715530b2f8edd927bb3af899669b64af91db65ca728ad56b0b6c7b96', 'c1f556d6eedf789bc202d6a84b3f39ca6664499609dbfeac9ee9e4bdfc180077')},
}


def _source_sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _contract_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _replace(source, old, new, count):
    if source.count(old) != count:
        raise ValueError('Reviewed frozen source replacement count changed: '+old)
    return source.replace(old, new)


def _rewrite_text(value, replacements):
    if isinstance(value, str):
        for old, new in replacements:
            value = value.replace(old, new)
        return value
    if isinstance(value, list):
        return [_rewrite_text(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _rewrite_text(item, replacements) for key, item in value.items()}
    return value


def _changed_paths(old, new, prefix=''):
    if old == new:
        return []
    if isinstance(old, dict) and isinstance(new, dict):
        return [path for key in sorted(old.keys() | new.keys())
                for path in (_changed_paths(old[key], new[key], prefix+'/'+key)
                    if key in old and key in new else [prefix+'/'+key])]
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        return [path for index, (a, b) in enumerate(zip(old, new))
                for path in _changed_paths(a, b, prefix+'/'+str(index))]
    return [prefix]


def adapt_payload(family, task_id, task, evaluator_sources):
    """Return ``(task, sources, report)`` for a new left/right bimanual variant.

    This does not authorize arbitrary source migration or adapt task geometry,
    robot reset, assets, physics, qualification or numeric scoring thresholds.
    It must not be used for the unchanged single-arm option. The new publisher
    owns robot installation, observer source/hash rebinding and episode dispatch.
    """
    pair = family, task_id
    if pair not in ALLOWED_TASKS:
        raise ValueError('Task is outside the reviewed single-agent bimanual allowlist')
    if not isinstance(task, dict) or task.get('id') != task_id:
        raise ValueError('Task identity differs from the requested adaptation')
    required = {'sequence_evaluator.py'} if pair == ('living_room', 'tower_building') else {'rules.py', 'sequence_evaluator.py'}
    if not isinstance(evaluator_sources, dict) or not required <= evaluator_sources.keys():
        raise ValueError('Expected the complete frozen evaluator source mapping')
    for name, source in evaluator_sources.items():
        path = PurePosixPath(name)
        if (not isinstance(source, str) or path.is_absolute() or '..' in path.parts
                or str(path) != name or not name.endswith('.py') or name.startswith('evaluator/')):
            raise ValueError('Evaluator source names must be relative Python paths inside evaluator/')
    for name, expected in SOURCE_PINS.get(pair, {}).items():
        accepted = (expected,) if isinstance(expected, str) else expected
        if name not in evaluator_sources or _source_sha(evaluator_sources[name]) not in accepted:
            raise ValueError('Unreviewed frozen evaluator source SHA256: '+name)
    result, sources = deepcopy(task), dict(evaluator_sources)
    replacements = []
    if pair in {('mechanical_assembly', name) for name in
                ('parts_sorting', 'peg_insertion', 'shaft_spacer_assembly')}:
        replacements = [('用单臂', '用机器人')]
        sources['rules.py'] = _replace(sources['rules.py'], *replacements[0], 3)
    elif pair == ('card_room', 'task0'):
        replacements = [('用单臂', '用机器人')]
        sources['rules.py'] = _replace(sources['rules.py'], *replacements[0], 1)
        if result.get('title_zh') != 'Task0 · 单臂随机三子摆放':
            raise ValueError('Unreviewed task0 title contract')
        result['title_zh'] = 'Task0 · 随机三子摆放'
    elif pair == ('living_room', 'water_pouring'):
        replacements = [('单臂抓起水壶', '用机器人抓起水壶')]
        sources['rules.py'] = _replace(sources['rules.py'], *replacements[0], 1)
    elif pair == ('mechanical_assembly', 'fastener_assembly'):
        config = result['evaluation']
        if config.get('hand_link_prefix') != 'right_' or config.get('robot_actor_prefix') != '/World/Robot/' or 'hand_link_prefixes' in config:
            raise ValueError('Expected the original frozen fastener hand namespace')
        config['hand_link_prefixes'] = list(HAND_PREFIXES)
        sources['sequence_evaluator.py'] = _replace(sources['sequence_evaluator.py'],
            "startswith(self.c['hand_link_prefix'])",
            "startswith(tuple(self.c['hand_link_prefixes']))", 2)
        sources['sequence_evaluator.py'] = _replace(sources['sequence_evaluator.py'],
            "        self.c = self.task['evaluation']\n",
            "        self.c = self.task['evaluation']\n"
            "        if (self.c.get('robot_actor_prefix') != '/World/Robot/'\n"
            "                or self.c.get('hand_link_prefixes') != ['left_', 'right_']):\n"
            "            raise ValueError('Exactly the declared left/right SHARPA namespaces are scored')\n", 1)
    elif pair == ('mechanical_assembly', 'gear_train_assembly'):
        # The whole-robot release and forbidden direct-gear/arm drive gates
        # already cover both arms. Only legitimate hand/tool holding namespaces
        # change; the opposing-normal and complete transmission-chain tests stay.
        sources['sequence_evaluator.py'] = _replace(sources['sequence_evaluator.py'],
            "other.rsplit('/', 1)[-1].startswith('right_')",
            "other.rsplit('/', 1)[-1].startswith(('left_', 'right_'))", 2)
    elif pair == ('living_room', 'remote_operation'):
        config = result['evaluation']
        original_fingers = [prefix for prefix in FINGER_PREFIXES if prefix.startswith('right_')]
        if (config.get('schema') not in ('living_remote_evaluation.v3', 'living_remote_evaluation.v4')
                or config.get('hand_link_prefixes') != ['right_']
                or config.get('finger_link_prefixes') != original_fingers):
            raise ValueError('Expected the original frozen television remote hand namespace')
        replacements = [
            ('用同一只手悬空握持遥控器', '可用左右手协作悬空握持遥控器'),
            ('用同一只手握持并依次', '可用左右手协作握持并依次'),
            ('Remote is off the table/holder and genuinely held by the same hand.',
             'Remote is off the table/holder and genuinely held by one or both declared hands.'),
            ('same-hand robot', 'bimanual robot'),
        ]
        for (old, new), count in zip(replacements, (3 if config['schema'] == 'living_remote_evaluation.v4' else 2, 1, 1, 2)):
            sources['rules.py'] = _replace(sources['rules.py'], old, new, count)
        sources['rules.py'] = _replace(sources['rules.py'], 'Frozen single-hand rules', 'Frozen bimanual rules', 1)
        sources['rules.py'] = _replace(sources['rules.py'],
            "'hand_link_prefixes': ['right_']", "'hand_link_prefixes': "+repr(HAND_PREFIXES), 1)
        sources['rules.py'] = _replace(sources['rules.py'],
            "'finger_link_prefixes': "+repr(original_fingers),
            "'finger_link_prefixes': "+repr(FINGER_PREFIXES), 1)
        source = sources['sequence_evaluator.py']
        source = _replace(source, "c['hand_link_prefixes'] != ['right_']",
                          "c['hand_link_prefixes'] != "+repr(HAND_PREFIXES), 1)
        source = _replace(source, 'Exactly the front right SHARPA hand is scored',
                          'Exactly the declared left/right SHARPA hands are scored', 1)
        source = _replace(source, "c['finger_link_prefixes'] != "+repr(original_fingers),
                          "c['finger_link_prefixes'] != "+repr(FINGER_PREFIXES), 1)
        source = _replace(source, "('right_hand_flange', 'right_hand_wrist')",
                          "('left_hand_flange', 'left_hand_wrist', 'right_hand_flange', 'right_hand_wrist')", 1)
        sources['sequence_evaluator.py'] = source
        config['hand_link_prefixes'] = list(HAND_PREFIXES)
        config['finger_link_prefixes'] = list(FINGER_PREFIXES)
    # Rewrite only the public contracts; provenance, limitations, physics and
    # private object geometry remain byte-for-byte equivalent as JSON values.
    for key in ('public_rules', 'public_goal'):
        if key in result and replacements:
            result[key] = _rewrite_text(result[key], replacements)
    if result.get('public_rules') != task.get('public_rules'):
        if 'public_rules_sha256' in result:
            result['public_rules_sha256'] = _contract_sha(result['public_rules'])
        if 'public_referee_contract_sha256' in result:
            result['public_referee_contract_sha256'] = _contract_sha(result['public_rules']['referee'])
        if 'public_rules_sha256' in result['evaluation']:
            result['evaluation']['public_rules_sha256'] = _contract_sha(result['public_rules'])
    changed_sources = {name: {'before_sha256': _source_sha(evaluator_sources[name]),
                             'after_sha256': _source_sha(source)}
                       for name, source in sources.items() if source != evaluator_sources[name]}
    report = {'schema': 'workroom_bimanual_scorer_adaptation.v1',
        'family': family, 'task_id': task_id, 'source_key_root': 'evaluator/',
        'changed_sources': changed_sources,
        # Freeze the exact selected source, not the evolving acceptance list.
        # This also preserves earlier releases' scalar pin/provenance bytes.
        'source_sha_pins_checked': {name: _source_sha(evaluator_sources[name])
                                   for name in SOURCE_PINS.get(pair, {})},
        'changed_task_paths': _changed_paths(task, result),
        'unchanged_sources': sorted(set(sources)-changed_sources.keys()),
        'numeric_thresholds_changed': False, 'object_physics_changed': False,
        'native_execution_performed': False,
        'observer_rebinding_required': 'Freeze all returned source hashes and the final task configuration together.',
        'episode_binding': ('frozen_sequence_evaluator.bind_episode_targets' if pair == ('card_room', 'task0')
            else 'frozen_rules.evaluation_configuration' if family == 'living_room' and task_id in ('remote_operation', 'water_pouring')
            else 'retain_base_frozen_contract')}
    return result, sources, report
