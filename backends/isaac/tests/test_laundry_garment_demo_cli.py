"""Engineering demos are explicitly selected and never implicit robot policies."""
import sys
import pytest
from scripts.validation import laundry_cloth_interactions as harness

@pytest.mark.parametrize('task', ['shirt_stacking', 'trousers_stacking'])
def test_garment_demo_requires_explicit_motion_profile(task, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'argv', ['record', '--case', task, '--output', str(tmp_path/'out')])
    with pytest.raises(SystemExit) as error:
        harness.main()
    assert error.value.code == 2
    assert not (tmp_path/'out').exists()

@pytest.mark.parametrize('task', ['shirt_stacking', 'trousers_stacking'])
def test_explicit_garment_demo_routes_to_engineering_harness(task, monkeypatch, tmp_path):
    seen=[]
    monkeypatch.setattr(harness, 'run_task', lambda args: seen.append(args) or 0)
    monkeypatch.setattr(sys, 'argv', ['record', '--case', task, '--output', str(tmp_path/'out'),
        '--garment-profile', 'folded_garment_leading_patches_v1', '--initialization', 'author-only'])
    with pytest.raises(SystemExit) as error:
        harness.main()
    assert error.value.code == 0
    assert seen[0].case == task and seen[0].initialization == 'author-only'

@pytest.mark.parametrize('task', ['towel_folding', 'lift_release'])
def test_garment_profile_cannot_change_historical_motion(task, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'argv', ['record', '--case', task, '--output', str(tmp_path/'out'),
        '--garment-profile', 'folded_garment_leading_patches_v1'])
    with pytest.raises(SystemExit) as error:
        harness.main()
    assert error.value.code == 2
