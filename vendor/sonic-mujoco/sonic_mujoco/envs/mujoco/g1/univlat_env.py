"""Runnable task approximations of Uni-VLaT's real-robot evaluation protocol.

The actors drive environmental interactions only. Robot motion always comes
from the supplied controller. Numerical contact tolerances are simulator choices.
"""

import operator
from copy import deepcopy
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from .g1_env import MujocoG1Env
from .sweep_env import OBJECT_NAMES
from .univlat_assets import SCENES, build_task

TASKS = {
    "back_tap_walking": (30.0, ("wait", "respond", "stop")),
    "table_sweeping": (60.0, ("sweep", "settle")),
    "basket_loading": (60.0, ("handover", "load", "takeaway", "release")),
    "human_robot_hugging": (30.0, ("approach", "embrace", "release")),
    "composed_cleanup": (180.0, ("sweep", "lift", "carry", "deposit")),
}


@dataclass(frozen=True)
class TaskState:
    stage: int = 0
    completed_stages: int = 0
    success: bool = False
    failed: bool = False
    elapsed: float = 0.0
    contact_force_n: float = 0.0
    object_count: int = 0
    failure_reason: str = ""


class MujocoG1UniVLaTEnv(MujocoG1Env):
    def __init__(self, task: str, timestep: float = 0.005) -> None:
        if task not in TASKS:
            raise ValueError(f"unknown Uni-VLaT task: {task}")
        self.task = task
        self.timeout, self.stages = TASKS[task]
        spec = build_task(task)
        super().__init__(SCENES / "empty.xml", timestep, model=spec.compile())
        self.model.vis.global_.azimuth = 135 if task == "back_tap_walking" else 40
        self.model.vis.global_.elevation = -23
        self.model.stat.center[:] = (0.5, 0.7 if task == "composed_cleanup" else 0, 0.8)
        self.model.stat.extent = 3.1 if task == "composed_cleanup" else 2.25
        self._torso = self.model.body("torso_link").id
        self._base_body_pos = self.model.body_pos.copy()
        self._base_geom_pos = self.model.geom_pos.copy()
        self._base_geom_size = self.model.geom_size.copy()
        self._start_time = None
        self._state = TaskState()
        self._reset_seed = None
        self._object_names = (
            OBJECT_NAMES
            if task == "table_sweeping"
            else tuple(f"rubbish_{i}" for i in range(4))
            if task == "composed_cleanup"
            else tuple(f"load_{i}" for i in range(6))
            if task == "basket_loading"
            else ()
        )
        self._object_ids = [self.model.body(name).id for name in self._object_names]
        self.task_metadata = {
            "schema_version": 1,
            "task": task,
            "source": "https://arxiv.org/html/2609.35450v2#A3",
            "implementation": "independent_mujoco_approximation",
            "geometry_calibration": "estimated_not_scanned",
            "human_model": "kinematic_contact_proxy",
            "timeout_s": self.timeout,
            "stages": list(self.stages),
            "thresholds": {
                "contact_n": 1.0,
                "fall_pelvis_height_m": 0.45,
                "min_torso_up_z": 0.5,
                "settle_s": 0.5,
                "back_response_deadline_s": 1.0,
                "walk_speed_m_s": 0.06,
                "stop_speed_m_s": 0.04,
                "hug_hold_s": 0.75,
            },
            "cleanup_stage_score": "first_three_stages_divided_by_three; success_requires_deposit",
        }

    def reset(self, seed: int | None = None) -> None:
        # Record the generated seed as well as explicit ones: reset(None) must
        # produce an episode that can later be reconstructed from its metadata.
        reset_seed = (
            int(np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0])
            if seed is None
            else operator.index(seed)
        )
        rng = np.random.default_rng(reset_seed)
        self.model.body_pos[:] = self._base_body_pos
        self.model.geom_pos[:] = self._base_geom_pos
        self.model.geom_size[:] = self._base_geom_size
        super().reset(reset_seed)
        self._reset_seed = reset_seed
        self._rng = rng
        self._start_time = None
        self._state = TaskState()
        self._held_since = None
        self._phase_time = 0.0
        self._last_contact = False
        self._edge_time = 0.0
        self._tap_count = 0
        self._tap_intervals = self._rng.integers(1, 5)
        self._loaded = 0
        self._anchor_basket = None
        self._load_order = self._rng.permutation(6)[: self._rng.integers(3, 7)]
        if self.task in ("table_sweeping", "basket_loading", "composed_cleanup"):
            # Leave room for the default extended hands, including raised tables.
            self.data.qpos[self._root_qpos_address] = -0.20
        if self.task == "table_sweeping":
            height = float(self._rng.uniform(0.75, 1.05))
            table = self.model.body("table").id
            delta = height - 0.77
            self.model.body_pos[table, 2] += delta
            # Keep the static table's feet on the floor as its top height changes.
            for geom_id in np.flatnonzero(self.model.geom_bodyid == table):
                z = self._base_geom_pos[geom_id, 2]
                if z < -0.6:
                    self.model.geom_pos[geom_id, 2] -= delta
                elif z < -0.2:
                    self.model.geom_pos[geom_id, 2] -= delta / 2
                    self.model.geom_size[geom_id, 2] += delta / 2
            for name, xy in zip(
                OBJECT_NAMES,
                (
                    (0.60, -0.16),
                    (0.98, -0.42),
                    (0.84, -0.16),
                    (0.68, -0.42),
                    (1.06, -0.16),
                ),
            ):
                self._place(
                    name,
                    (*np.add(xy, self._rng.uniform(-0.015, 0.015, 2)), height + 0.002),
                )
        else:
            for name in self._object_names:
                address = self._qpos(name)
                self.data.qpos[address : address + 2] += self._rng.uniform(
                    -0.015, 0.015, 2
                )
        # Recompute model constants after resizing the table; mj_setConst uses
        # qpos0 internally, so preserve this episode's randomized placement.
        initial_qpos = self.data.qpos.copy()
        mujoco.mj_setConst(self.model, self.data)
        self.data.qpos[:] = initial_qpos
        mujoco.mj_forward(self.model, self.data)
        self._episode_randomization = {
            "initial_object_poses": self._object_poses(),
        }
        if self.task == "table_sweeping":
            self._episode_randomization["table_height_m"] = height
        elif self.task == "back_tap_walking":
            self._episode_randomization["tap_cycles"] = int(self._tap_intervals)
        elif self.task == "basket_loading":
            self._episode_randomization["loading_order"] = [
                f"load_{i}" for i in self._load_order
            ]

    def episode_metadata(self) -> dict:
        """Return a detached context snapshot, not a resumable physics checkpoint.

        ``reset_seed`` reconstructs the episode's *initial* scene with this
        implementation. The task context identifies the phase and actor state
        when recording starts, which can be later than ``start_task()``.
        Poses are in the MuJoCo world frame, in meters and wxyz quaternions.
        """
        if self._reset_seed is None:
            raise RuntimeError(
                "reset the environment before requesting episode metadata"
            )
        elapsed = 0.0 if self._start_time is None else self.time - self._start_time
        root = self._root_qpos_address
        actors = {}
        for body in np.flatnonzero(self.model.body_mocapid >= 0):
            mocap = self.model.body_mocapid[body]
            actors[self.model.body(int(body)).name] = {
                "position_m": self.data.mocap_pos[mocap].tolist(),
                "quaternion_wxyz": self.data.mocap_quat[mocap].tolist(),
            }
        context = {
            "started": self._start_time is not None,
            "start_time_s": self._start_time,
            "simulation_time_s": self.time,
            "elapsed_s": elapsed,
            "stage_name": self.stages[self._state.stage],
            "state": asdict(self._state),
            "phase_start_elapsed_s": self._phase_time,
            "condition_held_since_elapsed_s": self._held_since,
            "robot_root_pose": {
                "position_m": self.data.qpos[root : root + 3].tolist(),
                "quaternion_wxyz": self.data.qpos[root + 3 : root + 7].tolist(),
            },
            "object_poses": self._object_poses(),
            "mocap_actors": actors,
        }
        if self.task == "back_tap_walking":
            context["observed_tap_count"] = self._tap_count
            context["contact_active"] = self._last_contact
            context["contact_edge_elapsed_s"] = self._edge_time
        elif self.task == "basket_loading":
            context["loaded_count"] = self._loaded
            context["basket_anchor_position_m"] = (
                None if self._anchor_basket is None else self._anchor_basket.tolist()
            )
        return {
            "schema_version": 1,
            "task": self.task,
            "reset_seed": self._reset_seed,
            "scene_randomization": deepcopy(self._episode_randomization),
            "task_context": context,
        }

    def _object_poses(self) -> dict:
        names = list(self._object_names)
        if self.task == "basket_loading":
            names.append("basket")
        elif self.task == "composed_cleanup":
            names.append("cleanup_box")
        poses = {}
        for name in names:
            address = self._qpos(name)
            poses[name] = {
                "position_m": self.data.qpos[address : address + 3].tolist(),
                "quaternion_wxyz": self.data.qpos[address + 3 : address + 7].tolist(),
            }
        return poses

    def start_task(self) -> None:
        if self._start_time is None:
            self._start_time = self.time

    def get_scene_state(self) -> TaskState:
        return self._state

    def task_observation(self) -> dict:
        return asdict(self._state)

    def is_success(self) -> bool:
        return self._state.success

    def _apply_command(self, command) -> None:
        super()._apply_command(command)
        if self._start_time is not None and not self._terminal:
            self._move_actor(self.time - self._start_time)

    def _after_physics_step(self) -> None:
        super()._after_physics_step()
        if self._start_time is None or self._terminal:
            return
        elapsed = self.time - self._start_time
        root_z = self.data.qpos[self._root_qpos_address + 2]
        up = self.data.xmat[self._torso].reshape(3, 3)[2, 2]
        if root_z < 0.45 or up < 0.5:
            self._fail("robot_fall", elapsed)
        elif elapsed > self.timeout:
            self._fail("timeout", elapsed)
        else:
            getattr(self, f"_evaluate_{self.task}")(elapsed)

    @property
    def _terminal(self):
        return self._state.success or self._state.failed

    def _report(
        self, elapsed, *, stage=None, complete=None, success=False, force=0.0, count=0
    ):
        self._state = TaskState(
            self._state.stage if stage is None else int(stage),
            self._state.completed_stages if complete is None else int(complete),
            bool(success),
            False,
            float(elapsed),
            float(force),
            int(count),
        )

    def _fail(self, reason, elapsed):
        self._state = TaskState(
            self._state.stage,
            self._state.completed_stages,
            False,
            True,
            elapsed,
            failure_reason=reason,
        )

    def _sustained(self, condition, elapsed, duration=0.5):
        if not condition:
            self._held_since = None
            return False
        if self._held_since is None:
            self._held_since = elapsed
        return elapsed - self._held_since >= duration

    def _advance(self, elapsed):
        self._held_since = None
        self._phase_time = elapsed
        self._report(
            elapsed, stage=self._state.stage + 1, complete=min(self._state.stage + 1, 3)
        )

    def _contact(self, other_names, *, back_only=False):
        targets = {self.model.body(name).id for name in other_names}
        force = np.zeros(6)
        regions = {"left": 0.0, "right": 0.0, "torso": 0.0}
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if min(contact.geom) < 0:
                continue
            b0, b1 = self.model.geom_bodyid[contact.geom]
            robot = b1 if b0 in targets else b0 if b1 in targets else -1
            if robot < 0:
                continue
            name = self.model.body(int(robot)).name
            region = (
                "torso"
                if name == "torso_link"
                else (
                    "left"
                    if name.startswith("left_")
                    else "right"
                    if name.startswith("right_")
                    else ""
                )
            )
            if region != "torso" and not any(
                s in name for s in ("shoulder", "elbow", "wrist", "hand")
            ):
                continue
            if back_only:
                local = self.data.xmat[self._torso].reshape(3, 3).T @ (
                    contact.pos - self.data.xpos[self._torso]
                )
                if region != "torso" or local[0] >= 0 or local[2] < 0.10:
                    continue
            if region:
                mujoco.mj_contactForce(self.model, self.data, index, force)
                regions[region] += max(0.0, float(force[0]))
        return regions

    def _move_actor(self, elapsed):
        if self.task == "back_tap_walking":
            active = (
                1.0 <= elapsed < 1 + 4 * self._tap_intervals and (elapsed - 1) % 4 < 2
            )
            phase = (elapsed - 1) % 4
            blend = min(1.0, phase / 0.25, (2 - phase) / 0.25) if active else 0.0
            local = np.array((-0.27 + max(0.0, blend) * 0.167, 0, 0.22))
            rotation = self.data.xmat[self._torso].reshape(3, 3)
            pos = self.data.xpos[self._torso] + rotation @ local
            self._mocap("back_palm", pos, self.data.xquat[self._torso])
            root = self.data.qpos[self._root_qpos_address : self._root_qpos_address + 3]
            self._mocap("participant", (root[0] - 0.65, root[1] - 0.55, 0))
        elif self.task == "human_robot_hugging" and self._state.stage == 2:
            self._mocap(
                "participant",
                (1.0 + min(0.6, (elapsed - self._phase_time) * 0.25), 0, 0),
            )
        elif self.task == "basket_loading":
            basket = self.data.xpos[self.model.body("basket").id]
            if self._state.stage == 0:
                self._mocap(
                    "handover_support", (0.66, 0, 0.735 + min(0.18, elapsed * 0.06))
                )
            elif self._state.stage == 1:
                self._mocap("handover_support", (1.5, -0.7, 0.65))
                if self._loaded < len(
                    self._load_order
                ) and elapsed - self._phase_time >= 2 * (self._loaded + 1):
                    obj = self._load_order[self._loaded]
                    self._place(
                        f"load_{obj}",
                        basket + np.array((0, (-1) ** self._loaded * 0.1, 0.30)),
                    )
                    self._loaded += 1
                    mujoco.mj_forward(self.model, self.data)
            elif self._state.stage >= 2:
                # A visible support approaches under the basket; no weld or robot control.
                offset = min(0.18, (elapsed - self._phase_time) * 0.04)
                self._mocap(
                    "handover_support",
                    (basket[0] + offset, basket[1], basket[2] - 0.022),
                )

    def _evaluate_back_tap_walking(self, elapsed):
        force = self._contact(("back_palm",), back_only=True)["torso"]
        contact = force >= 1.0
        speed = self.data.qvel[self._root_dof_address]
        # Contact chatter shorter than 50ms does not create a new command edge.
        if contact != self._last_contact and elapsed - self._edge_time >= 0.05:
            self._last_contact = contact
            self._edge_time = elapsed
            self._tap_count += int(contact)
        responding = (self._last_contact and speed > 0.06) or (
            not self._last_contact and abs(speed) < 0.04
        )
        if not responding and elapsed - self._edge_time > 1.0:
            self._fail(
                "late_walk_response" if self._last_contact else "late_stop_response",
                elapsed,
            )
            return
        done = (
            elapsed > 2 + 4 * self._tap_intervals
            and self._tap_count >= self._tap_intervals
            and not self._last_contact
            and abs(speed) < 0.04
        )
        self._report(
            elapsed,
            stage=1 if contact else 2 if self._tap_count else 0,
            complete=3 if done else 0,
            success=done,
            force=force,
            count=self._tap_count,
        )

    def _evaluate_table_sweeping(self, elapsed):
        table = self.model.body("table").id
        points = self.data.xpos[self._object_ids] - self.data.xpos[table]
        if (
            np.any(points[:, 2] < -0.06)
            or np.any(np.abs(points[:, 0]) > 0.49)
            or np.any(np.abs(points[:, 1]) > 0.73)
        ):
            self._fail("object_fell_from_table", elapsed)
            return
        # Entire footprints must clear the divider, not only the centers.
        radii = np.array((0.045, 0.060, 0.072, 0.091, 0.08))
        crossed = points[:, 1] - radii > 0.012
        force = sum(self._contact(("table",)).values())
        if force > 250:
            self._fail("excessive_table_contact", elapsed)
            return
        done = self._sustained(bool(crossed.all()), elapsed)
        self._report(
            elapsed,
            stage=int(crossed.all()),
            complete=2 if done else 0,
            success=done,
            count=int(crossed.sum()),
            force=force,
        )

    def _evaluate_human_robot_hugging(self, elapsed):
        contacts = self._contact(("participant",))
        bilateral = contacts["left"] > 1 and contacts["right"] > 1
        force = sum(contacts.values())
        if force > 120:
            self._fail("excessive_hug_contact", elapsed)
            return
        if (
            self._state.stage == 0
            and bilateral
            or self._state.stage == 1
            and self._sustained(bilateral, elapsed, 0.75)
        ):
            self._advance(elapsed)
        elif self._state.stage == 2:
            quiet = (
                np.linalg.norm(
                    self.data.qvel[self._root_dof_address : self._root_dof_address + 3]
                )
                < 0.10
            )
            if self._sustained(force < 1 and quiet, elapsed):
                self._report(elapsed, complete=3, success=True, force=force)
                return
        self._report(elapsed, force=force)

    def _evaluate_basket_loading(self, elapsed):
        contacts = self._contact(("basket",))
        bilateral = contacts["left"] > 1 and contacts["right"] > 1
        basket = self.model.body("basket").id
        pos = self.data.xpos[basket]
        if pos[2] < 0.5 or self.data.xmat[basket].reshape(3, 3)[2, 2] < 0.7:
            self._fail("basket_dropped_or_tipped", elapsed)
            return
        loaded_names = [f"load_{i}" for i in self._load_order[: self._loaded]]
        loaded_ids = [self.model.body(name).id for name in loaded_names]
        local = (self.data.xpos[loaded_ids] - pos) @ self.data.xmat[basket].reshape(
            3, 3
        )
        contents_inside = bool(loaded_ids) and np.all(
            (np.abs(local[:, 0]) < 0.212)
            & (np.abs(local[:, 1]) < 0.282)
            & (local[:, 2] > -0.01)
        )
        if loaded_ids and np.any(self.data.xpos[loaded_ids, 2] < pos[2] - 0.08):
            self._fail("basket_contents_dropped", elapsed)
            return
        if self._state.stage == 0 and self._sustained(
            bilateral and pos[2] > 0.85, elapsed
        ):
            self._anchor_basket = pos.copy()
            self._advance(elapsed)
        elif self._state.stage == 1:
            if np.linalg.norm(pos[:2] - self._anchor_basket[:2]) > 0.35:
                self._fail("basket_displaced", elapsed)
                return
            loading_finished = self._loaded == len(self._load_order)
            settled = True
            for name in loaded_names:
                address = self.model.joint(f"{name}_joint").dofadr[0]
                settled &= np.linalg.norm(self.data.qvel[address : address + 3]) < 0.15
            if self._sustained(
                loading_finished and contents_inside and settled, elapsed
            ):
                self._advance(elapsed)
        elif self._state.stage == 2:
            supported = sum(self._contact_between("basket", "handover_support")) > 1
            if self._sustained(
                contents_inside and sum(contacts.values()) < 1 and supported, elapsed
            ):
                self._report(
                    elapsed, stage=3, complete=3, success=True, count=self._loaded
                )
                return
        self._report(elapsed, count=self._loaded, force=sum(contacts.values()))

    def _evaluate_composed_cleanup(self, elapsed):
        box = self.model.body("cleanup_box").id
        contacts = self._contact(("cleanup_box",))
        bilateral = contacts["left"] > 1 and contacts["right"] > 1
        inside = [
            self._body_in_site(name, "cleanup_box_interior")
            for name in self._object_names
        ]
        rubbish_dropped = any(
            self.data.xpos[body, 2] < 0.15
            and not self._body_in_site(name, "waste_bin_interior")
            for name, body in zip(self._object_names, self._object_ids)
        )
        if rubbish_dropped or (self.data.xpos[box, 2] < 0.1 and not self._box_in_bin()):
            self._fail("cleanup_object_dropped", elapsed)
            return
        stage = self._state.stage
        if (
            stage == 0
            and self._sustained(all(inside), elapsed)
            or stage == 1
            and self._sustained(
                all(inside) and bilateral and self.data.xpos[box, 2] > 0.74, elapsed
            )
        ):
            self._advance(elapsed)
        elif stage == 2:
            bin_pos = self.data.xpos[self.model.body("waste_bin").id]
            near = np.linalg.norm(self.data.xpos[box, :2] - bin_pos[:2]) < 0.7
            if self._sustained(
                near and all(inside) and bilateral and self.data.xpos[box, 2] > 0.74,
                elapsed,
            ):
                self._advance(elapsed)
        elif stage == 3:
            deposited = (
                self._box_in_bin() and all(inside) and sum(contacts.values()) < 1
            )
            address = self.model.joint("cleanup_box_joint").dofadr[0]
            settled = np.linalg.norm(self.data.qvel[address : address + 6]) < 0.15
            if self._sustained(deposited and settled, elapsed):
                self._report(elapsed, complete=3, success=True, count=sum(inside))
                return
        self._report(elapsed, count=sum(inside), force=sum(contacts.values()))

    def _body_in_site(self, name, site_name):
        site = self.model.site(site_name).id
        # Cleanup rubbish consists of boxes: test all corners, including rotation.
        geom = self.model.geom(f"{name}_geom").id
        half = self.model.geom_size[geom]
        corners = np.array(
            [
                (x, y, z)
                for x in (-half[0], half[0])
                for y in (-half[1], half[1])
                for z in (-half[2], half[2])
            ]
        )
        world = (
            corners @ self.data.geom_xmat[geom].reshape(3, 3).T
            + self.data.geom_xpos[geom]
        )
        local = (world - self.data.site_xpos[site]) @ self.data.site_xmat[site].reshape(
            3, 3
        )
        return bool(np.all(np.abs(local) <= self.model.site_size[site] + 0.003))

    def _box_in_bin(self):
        body = self.model.body("cleanup_box").id
        site = self.model.site("waste_bin_interior").id
        corners = np.array(
            [
                (x, y, z)
                for x in (-0.278, 0.278)
                for y in (-0.298, 0.298)
                for z in (-0.008, 0.22)
            ]
        )
        world = corners @ self.data.xmat[body].reshape(3, 3).T + self.data.xpos[body]
        local = (world - self.data.site_xpos[site]) @ self.data.site_xmat[site].reshape(
            3, 3
        )
        return bool(np.all(np.abs(local) <= self.model.site_size[site] + 0.005))

    def _contact_between(self, first, second):
        ids = {self.model.body(first).id, self.model.body(second).id}
        force = np.zeros(6)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if (
                min(contact.geom) >= 0
                and set(self.model.geom_bodyid[contact.geom]) == ids
            ):
                mujoco.mj_contactForce(self.model, self.data, i, force)
                yield max(0.0, float(force[0]))

    def _qpos(self, body):
        return int(self.model.joint(f"{body}_joint").qposadr[0])

    def _place(self, body, position):
        address = self._qpos(body)
        self.data.qpos[address : address + 3] = position
        self.data.qpos[address + 3 : address + 7] = (1, 0, 0, 0)
        velocity = int(self.model.joint(f"{body}_joint").dofadr[0])
        self.data.qvel[velocity : velocity + 6] = 0

    def _mocap(self, name, position, quaternion=(1, 0, 0, 0)):
        index = self.model.body_mocapid[self.model.body(name).id]
        self.data.mocap_pos[index] = position
        self.data.mocap_quat[index] = quaternion
