"""Explicit robot-revision practice runtime, retaining the original action gates."""
from pathlib import Path

from scenes.workrooms.card_room.practice.runtime import PracticeObserver, PracticeRuntime
from .practice import RobotRevisionPracticeReferee


class RobotRevisionPracticeRuntime(PracticeRuntime):
    """A new identity from construction, with verified legacy scorer delegation.

    Action authorization, ACKs, contact observation, termination and receipt
    writing are inherited unchanged. There is no post-construction identity
    replacement and no task-object or articulation state write.
    """
    def __init__(self, env, scene, task, output, *, evaluator_factory=None):
        self.env, self.task = env, task
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.evaluator = RobotRevisionPracticeReferee(task)
        self.side = self.evaluator.public['active_side']
        if scene.get('card_practice_task') != {'schema': 'card_rsi_practice_native.v1',
                'task_id': task['id'], 'active_side': self.side,
                'private_case_sha256': self.evaluator.case['private_case_sha256']}:
            raise ValueError('Revised practice native scene/case/side binding mismatch')
        self.names = tuple(env.actuated_joint_names)
        self.owned = frozenset(n for n in self.names if n.startswith((self.side+'_', 'fr3_'+self.side+'_')))
        if len(self.owned) != 29 or len(self.names) != 58:
            raise ValueError('Revised practice retains the original two 29-DOF ownership partitions')
        self.targets = {n: float(env._targets[0, env.joint_names.index(n)]) for n in self.names}
        self.observer = PracticeObserver(env, self.evaluator, task['asset_metadata'],
            frame_jsonl=self.output/'native_frames.jsonl').attach()
        self.start_native_step = self.observer.last_step
        self.start_simulation_time_s = self.observer.last_time
        self.closed = False


def build_runtime(env, scene, task, output, *, evaluator_factory):
    return RobotRevisionPracticeRuntime(env, scene, task, output, evaluator_factory=evaluator_factory)
