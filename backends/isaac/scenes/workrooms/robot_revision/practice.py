"""Identity-only practice derivation over verified historical reset/goal cases."""
from copy import deepcopy

from scenes.workrooms.card_room.practice.cases import verify_case
from .inputs import canonical_hash, derive_inputs

LINEAGE_SCHEMA = 'robot_revision_practice_lineage.v1'


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Practice robot revision requires an exact digest')
    return value


def derive_case(original, metadata, *, revision_sha256, formal_release_sha256):
    """Verify the entire old case before rebinding its robot/formal identity."""
    original = verify_case(original, metadata)
    _digest(revision_sha256); _digest(formal_release_sha256)
    if formal_release_sha256 == original['public']['parent_release_sha256']:
        raise ValueError('Practice revision must select a new formal robot release')
    case = deepcopy(original)
    public = case['public']
    public['parent_release_sha256'] = formal_release_sha256
    public['case_sha256'] = canonical_hash({k:v for k,v in public.items() if k != 'case_sha256'})
    case['robot_revision'] = {'schema': LINEAGE_SCHEMA, 'revision_sha256': revision_sha256,
                             'parent_private_case_sha256': original['private_case_sha256']}
    case['case_fingerprint'] = canonical_hash({'parent_case_fingerprint': original['case_fingerprint'],
        'robot_revision_sha256': revision_sha256, 'formal_release_sha256': formal_release_sha256})
    case['private_case_sha256'] = canonical_hash({k:v for k,v in case.items() if k != 'private_case_sha256'})
    return case


def derive_practice_inputs(scene, task, initial, revision, *, formal_release):
    """Return new externally consistent identities and a private legacy view.

    The publisher validates the new formal release's complete input pins. Its
    reference remains private; only the existing public parent digest changes.
    """
    if (not isinstance(formal_release, dict) or set(formal_release) != {'path', 'manifest_sha256'}
            or not isinstance(formal_release['path'], str) or not formal_release['path'].startswith('/')):
        raise ValueError('Practice requires its explicit revised formal release pin')
    original = task['practice_case']
    case = derive_case(original, task['asset_metadata'], revision_sha256=canonical_hash(revision),
                       formal_release_sha256=formal_release['manifest_sha256'])
    plain = deepcopy(task); plain.pop('practice_case')
    s, t, i = derive_inputs(scene, plain, initial, revision)
    t['practice_case'] = case
    t['robot_revision_practice'] = {'schema': LINEAGE_SCHEMA,
        'revision_sha256': canonical_hash(revision), 'parent_case': deepcopy(original),
        'formal_release': deepcopy(formal_release)}
    expected = {'schema': 'card_rsi_practice_native.v1', 'task_id': task['id'],
        'active_side': original['public']['active_side'], 'private_case_sha256': original['private_case_sha256']}
    if scene.get('card_practice_task') != expected:
        raise ValueError('Original practice scene/case binding differs')
    s['card_practice_task'] = {**expected, 'private_case_sha256': case['private_case_sha256']}
    return s, t, i


def verified_legacy_view(task):
    """Revalidate both identities; never accept a caller-supplied unverified view."""
    lineage = task.get('robot_revision_practice')
    if (not isinstance(lineage, dict) or set(lineage) != {'schema', 'revision_sha256', 'parent_case', 'formal_release'}
            or lineage['schema'] != LINEAGE_SCHEMA):
        raise ValueError('Practice revision lineage is incomplete')
    reference = lineage['formal_release']
    if (not isinstance(reference, dict) or set(reference) != {'path', 'manifest_sha256'}
            or not isinstance(reference['path'], str) or not reference['path'].startswith('/')):
        raise ValueError('Practice formal-parent reference is invalid')
    expected = derive_case(lineage['parent_case'], task['asset_metadata'],
        revision_sha256=lineage['revision_sha256'], formal_release_sha256=reference['manifest_sha256'])
    if canonical_hash(task['practice_case']) != canonical_hash(expected):
        raise ValueError('Practice derivation changed more than its verified robot identity')
    legacy = deepcopy(task)
    legacy.pop('robot_revision_practice')
    legacy['practice_case'] = deepcopy(lineage['parent_case'])
    return legacy


class RobotRevisionPracticeReferee:
    """Expose the new identity while delegating unchanged physical adjudication."""
    def __init__(self, task):
        from scenes.workrooms.card_room.practice.referee import PracticeReferee
        legacy = verified_legacy_view(task)
        self.task = deepcopy(task)
        self.case = deepcopy(task['practice_case'])
        self.public = self.case['public']
        self._legacy = PracticeReferee(legacy)

    def __getattr__(self, name):
        return getattr(self._legacy, name)

    @property
    def contact_rows(self):
        return self._legacy.contact_rows

    @contact_rows.setter
    def contact_rows(self, rows):
        self._legacy.contact_rows = rows

    def observe(self, frame):
        return self._legacy.observe(frame)

    def score(self):
        return self._legacy.score()

    def interrupt(self, reason):
        return self._legacy.interrupt(reason)


def evaluator_factory(task):
    return RobotRevisionPracticeReferee(task)
