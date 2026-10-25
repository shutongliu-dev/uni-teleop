"""Actionable held-input reasons without printing the same fault every frame."""
from types import SimpleNamespace
import json

from scripts.teleop.diagnostics import HoldReporter, control_issues


def test_console_reports_joint_values_on_change_and_throttles_repeats(capsys):
    reporter=HoldReporter()
    issue={'channel':'arm','code':'joint_limit','joint':'fr3_joint4',
           'target_rad':-3.4,'lower_rad':-3.08,'upper_rad':-.12,'reason':'outside range'}
    reporter.update([],now_ns=0)
    assert not capsys.readouterr().out
    reporter.update([issue],now_ns=1)
    printed=json.loads(capsys.readouterr().out)
    assert printed['event']=='teleop_hold' and printed['issues'][0]['target_rad']==-3.4
    reporter.update([issue | {'target_rad':-3.5}],now_ns=100_000_000)
    assert not capsys.readouterr().out
    reporter.update([issue | {'target_rad':-3.6}],now_ns=5_000_000_001)
    assert json.loads(capsys.readouterr().out)['issues'][0]['target_rad']==-3.6
    reporter.update([issue | {'joint':'fr3_joint2'}],now_ns=5_000_000_002)
    assert json.loads(capsys.readouterr().out)['issues'][0]['joint']=='fr3_joint2'
    reporter.update([],now_ns=5_000_000_003)
    assert json.loads(capsys.readouterr().out)['event']=='teleop_hold_cleared'


def test_arm_glove_and_physics_reasons_are_distinct():
    mapper=SimpleNamespace(issue={'code':'leader_stale','reason':'stale','age_ms':180.},
        diagnostics={'arm':'same arm text','hand':'MANUS expired; holding','fr3_joint1':'alignment error'})
    issues=control_issues(mapper,SimpleNamespace(trigger={'reason':'penetration'}))
    assert [v['channel'] for v in issues]==['arm','hand','physics']
    assert [v['code'] for v in issues]==['leader_stale','input_hold','contact_guard']
