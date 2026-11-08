"""Synthetic native-frame counterexamples, not physical manipulation claims."""
from dataclasses import replace
import math

import pytest

from test_card_practice_referee import make, baseline, advance, target_objects
from test_card_room_expansion_referee import Rig


CASES = [(task, side) for task in ('polyomino_territory_practice',
    'barrier_race_practice', 'memory_pairs_practice') for side in ('left', 'right')]
CASES += [('polyomino_territory', 'left'), ('barrier_race', 'left')]


class Trace:
    def __init__(self, task, side):
        self.practice = task.endswith('_practice')
        if self.practice:
            stratum = {'polyomino_territory_practice': 'mono_translate',
                'barrier_race_practice': 'pawn_step', 'memory_pairs_practice': 'restore'}[task]
            self.ref = make(task, stratum, side)
            self.frame = baseline(self.ref)
            self.ref.observe(self.frame)
            self.name = self.ref.stage['targets'][0]['object_id']
            self.goal = target_objects(self.ref, self.frame, contact=False)[self.name]
            self.player = self.ref.player
        else:
            self.rig = Rig(task, baseline=False)
            self.rig.dt = .0005
            for name, obj in self.rig.objects.items():
                self.rig.modify(name, deck_support_impulse_ns=.1*9.81*self.rig.dt)
            self.rig.tick()
            self.ref = self.rig.match
            self.player = self.ref.current_player
            if task == 'polyomino_territory':
                self.name = 'black_mono'
                self.rig.intent(type='place_piece', piece_id=self.name, anchor=[0, 0])
                goal = (.52, -.13, .817)
            else:
                self.name = 'black_pawn'
                self.rig.intent(type='move_pawn', destination=[2, 1])
                goal = (.65, -.075, .8485)
            self.goal = replace(self.object, position_m=goal)
        self.start = self.object

    @property
    def object(self):
        return self.frame.objects[self.name] if self.practice else self.rig.objects[self.name]

    @property
    def credited(self):
        return self.name in (self.ref.motion_seen if self.practice else self.ref._motion_seen)

    def tick(self, pose=None, contact='none'):
        pose = pose or self.object
        impulses = {p: float(p == self.player and contact != 'none') for p in ('player_0', 'player_1')}
        nonhand = {p: float(p == self.player and contact == 'arm') for p in impulses}
        obj = replace(self.object, position_m=pose.position_m, quaternion_wxyz=pose.quaternion_wxyz,
            robot_contact_impulses_ns=impulses, robot_nonhand_contact_impulses_ns=nonhand)
        if self.practice:
            self.frame = advance(self.frame, **{self.name: obj})
            return self.ref.observe(self.frame)
        self.rig.objects[self.name] = obj
        return self.rig.tick()

    def move(self, goal, contact, steps=200):
        start = self.object
        for i in range(1, steps+1):
            u = i/steps
            position = tuple(a+(b-a)*u for a, b in zip(start.position_m, goal.position_m))
            q = tuple(a+(b-a)*u for a, b in zip(start.quaternion_wxyz, goal.quaternion_wxyz))
            norm = math.sqrt(sum(v*v for v in q))
            self.tick(replace(goal, position_m=position, quaternion_wxyz=tuple(v/norm for v in q)), contact)

    def settle(self):
        for _ in range(601):
            self.tick()
        return self.ref.score()['success'] if self.practice else self.ref.turn_index > 0


@pytest.mark.parametrize('task,side', CASES)
@pytest.mark.parametrize('fault', ['arm_before_hand', 'hand_before_arm', 'gap', 'jitter',
    'angular_jitter', 'separate_small_segments'])
def test_motion_without_hand_cannot_be_credited_by_an_unrelated_touch(task, side, fault):
    trace = Trace(task, side)
    if fault == 'hand_before_arm':
        trace.tick(contact='hand')
        trace.tick()
        trace.move(trace.goal, 'arm')
    else:
        if fault == 'gap':
            small = replace(trace.object, position_m=tuple(v+(.001 if i == 0 else 0.)
                for i, v in enumerate(trace.object.position_m)))
            trace.move(small, 'hand', steps=20)
            trace.tick()
        if fault == 'separate_small_segments':
            trace.move(replace(trace.goal, position_m=(trace.goal.position_m[0]-.006,
                *trace.goal.position_m[1:])), 'arm')
            for dx in (-.004, -.002, 0.):
                trace.move(replace(trace.goal, position_m=(trace.goal.position_m[0]+dx,
                    *trace.goal.position_m[1:])), 'hand', steps=20)
                trace.tick()
        else:
            trace.move(trace.goal, 'arm')
        trace.tick(contact='hand')
        if fault == 'jitter':
            # Long travelled path, but no contact-linked net 5 mm / 10 deg.
            for i in range(60):
                pose = replace(trace.goal, position_m=(trace.goal.position_m[0]+(.001 if i % 2 else -.001),
                    *trace.goal.position_m[1:]))
                trace.tick(pose, 'hand')
            trace.tick(trace.goal, 'hand')
        if fault == 'angular_jitter':
            w, x, y, z = trace.goal.quaternion_wxyz
            for i in range(60):
                half = math.radians(2 if i % 2 else -2)/2
                c, s = math.cos(half), math.sin(half)
                trace.tick(replace(trace.goal, quaternion_wxyz=(c*w-s*z, c*x-s*y,
                    c*y+s*x, c*z+s*w)), 'hand')
            trace.tick(trace.goal, 'hand')
    assert not trace.credited
    assert not trace.settle()


@pytest.mark.parametrize('task,side', CASES)
@pytest.mark.parametrize('regrasp', [False, True])
def test_continuous_small_hand_steps_and_regrasp_keep_released_success(task, side, regrasp):
    trace = Trace(task, side)
    if regrasp:
        trace.tick(contact='hand')
        trace.tick()
        # An unsuccessful earlier touch is recoverable, never a permanent fault.
    trace.move(trace.goal, 'hand')
    assert trace.credited
    assert trace.settle()


def test_cancelled_intent_cannot_reuse_earlier_hand_motion():
    trace = Trace('polyomino_territory', 'left')
    trace.move(trace.goal, 'hand')
    assert trace.credited
    trace.ref.cancel_intent(trace.player)
    trace.rig.intent(type='place_piece', piece_id=trace.name, anchor=[0, 0])
    trace.tick(contact='hand')
    assert not trace.credited
    assert not trace.settle()
