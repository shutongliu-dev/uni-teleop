"""A frozen evaluator must never reuse a different release's relative helper."""
import json

import pytest

from bench.isaacsim.release import digest, load_evaluator


def make_release(root, marker):
    (root/'evaluator').mkdir(parents=True)
    evaluator = root/'evaluator/sequence_evaluator.py'
    evaluator.write_text('from .humanlike import MARKER\nclass SequenceEvaluator:\n    marker = MARKER\n')
    helper = root/'evaluator/humanlike.py'
    helper.write_text(f'MARKER = {marker!r}\n')
    profile = root/'physics_profile.json'
    profile.write_text('{}\n')
    manifest = {
        'schema_version': 'isaacsim-simple-manipulation-release-v1',
        'source_release': None, 'original_release_modified': False,
        'files': {str(path.relative_to(root)): {'sha256':digest(path)} for path in (evaluator, helper)},
        'physics_profile': profile.name, 'physics_profile_sha256':digest(profile),
    }
    (root/'release_manifest.json').write_text(json.dumps(manifest))
    return evaluator, helper


def test_same_evaluator_with_different_frozen_helpers_never_shares_cached_package(tmp_path):
    first, second = tmp_path/'release_a', tmp_path/'release_b'
    first_evaluator, _ = make_release(first, 'first_constraints')
    second_evaluator, _ = make_release(second, 'second_constraints')
    assert digest(first_evaluator) == digest(second_evaluator)
    a, b = load_evaluator(first), load_evaluator(second)
    assert a.marker == 'first_constraints' and b.marker == 'second_constraints'
    assert a is not b and a.__module__ != b.__module__
    assert load_evaluator(first) is a and load_evaluator(second) is b


def test_cached_evaluator_does_not_hide_tampered_frozen_helper(tmp_path):
    root = tmp_path/'release'
    _, helper = make_release(root, 'original')
    assert load_evaluator(root).marker == 'original'
    helper.write_text("MARKER = 'changed_without_release_version'\n")
    with pytest.raises(ValueError, match='Frozen input changed'):
        load_evaluator(root)
