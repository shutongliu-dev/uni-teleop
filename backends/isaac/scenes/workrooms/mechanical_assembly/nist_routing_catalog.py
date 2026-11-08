"""Publicly identifiable routing tasks; preserve prior release qualifications."""
from copy import deepcopy

from bench.paths import DATA_ROOT

RELEASES = {'nist_cable_routing': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/nist_cable_routing/desktop_v2',
                        'manifest_sha256': '4b765f28ed389f6243d810ec4eed844c30698b1e64761a86ec4a191112523f6e'},
 'nist_wire_harness': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly/nist_wire_harness/desktop_v2',
                       'manifest_sha256': 'aabb4765330fba869a7ee7475c03f29a3814fcd041757056dff196e2fb8ef0a4'}}
PUBLICATION_EVIDENCE = {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_routing_public_20261021/publication01.json', 'sha256': 'd89df6a54ddb63460b76c4a398ea5178a3d19b295b1099183ae2295e4a056edb'}

NATIVE_INTERFACE_EVIDENCE = {'nist_cable_routing': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_routing_public_20261021/task8_http01/qualification.json',
                        'sha256': 'fcfcf1de9948e348fb7bdae8ce918b3a791267cad7c9a3c57111352be652c0ed',
                        'release_manifest_sha256': '4b765f28ed389f6243d810ec4eed844c30698b1e64761a86ec4a191112523f6e',
                        'passed': True,
                        'independent_review': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_routing_public_20261021/task8_http01_independent01.json',
                                               'sha256': '28f337e41ef0c5f17685f77a8c67e54354edf55115cae4c719eac590b6665cab'},
                        'scope': 'This desktop_v2 release only: live native '
                                 'HTTP discovery, Agent get_task/static '
                                 'context, four RGB-D cameras, read-only IK, '
                                 'wrist micro-motion/return/hold and '
                                 'idempotency. No hand-object assembly, '
                                 'reversed placement or removal execution.'},
 'nist_wire_harness': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_routing_public_20261021/task9_http01/qualification.json',
                       'sha256': '8e9e11924fb9733a77b0e0df5b7f7f0144d81f178ec10fc9300ddab74a923196',
                       'release_manifest_sha256': 'aabb4765330fba869a7ee7475c03f29a3814fcd041757056dff196e2fb8ef0a4',
                       'passed': True,
                       'independent_review': {'path': f'{DATA_ROOT}/datasets/fr3_sharpa_bench/development/mechanical_routing_public_20261021/task9_http01_independent01.json',
                                              'sha256': '7c43b0e9534db347d54148fb7eea2491a38e2b01eca2d7b0edd34f4997fa4de2'},
                       'scope': 'This desktop_v2 release only: live native '
                                'HTTP discovery, Agent get_task/static '
                                'context, four RGB-D cameras, read-only IK, '
                                'wrist micro-motion/return/hold and '
                                'idempotency. No hand-object assembly, '
                                'reversed placement or removal execution.'}}


def bind_public_routing_release(task):
    result = deepcopy(task)
    result['previous_routing_release'] = {
        'source_release': deepcopy(result['source_release']),
        'qualification': deepcopy(result['qualification']),
    }
    result['source_release'] = deepcopy(RELEASES[result['task_id']])
    result['qualification'] = {
        'release_status': 'prepared_unvalidated',
        'native_preflight': True, 'agent_bench_interface': True,
        'manipulation': False, 'physics_accepted': False,
        'task_policy_success_claimed': False,
        'scene_contact_evidence': [],
        'publication_evidence': deepcopy(PUBLICATION_EVIDENCE),
        'native_interface_evidence': deepcopy(NATIVE_INTERFACE_EVIDENCE[result['task_id']]),
        'scope': 'Task8 explicit observable route; Task9 whole-lead endpoint equivalence. Physical scene, assets, reset, robot and shared control are identical to desktop_v1. Rules/evaluator are separately frozen; historical native evidence does not qualify this release.',
    }
    return result
