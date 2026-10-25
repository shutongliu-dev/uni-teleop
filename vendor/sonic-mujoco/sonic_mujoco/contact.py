from dataclasses import dataclass, field, replace

import mujoco
import numpy as np

MAX_CONTACTS = 16


@dataclass(frozen=True, slots=True)
class ContactFrame:
    """Robot-to-scene contacts accumulated over one control interval."""

    robot_body_id: np.ndarray
    other_body_id: np.ndarray
    position: np.ndarray
    normal_force: np.ndarray
    tangent_force: np.ndarray
    normal_impulse: np.ndarray
    sample_count: np.ndarray
    count: int


@dataclass(slots=True)
class _ContactValue:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    normal_force: float = 0.0
    tangent_force: float = 0.0
    normal_impulse: float = 0.0
    sample_count: int = 0


class ContactRecorder:
    """Aggregate MuJoCo contacts at physics rate into fixed control frames."""

    def __init__(self, model: mujoco.MjModel, robot_root: str = "pelvis") -> None:
        self.model = model
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_root)
        if root < 0:
            raise ValueError(f"missing robot root body: {robot_root}")
        self.robot_body_ids = frozenset(
            body_id
            for body_id in range(model.nbody)
            if self._is_descendant(body_id, root)
        )
        self.body_names = tuple(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            or f"body_{body_id}"
            for body_id in range(model.nbody)
        )
        self._flex_body_ids = tuple(
            self._flex_root_body(flex_id) for flex_id in range(model.nflex)
        )
        self._pairs: dict[tuple[int, int], _ContactValue] = {}
        self.last_frame = self._empty_frame()

    def begin(self) -> None:
        self._pairs.clear()

    def reset(self) -> None:
        """Discard the active interval and the previously finished frame."""

        self.begin()
        self.last_frame = self._empty_frame()

    def update(self, data: mujoco.MjData) -> None:
        force = np.zeros(6, dtype=np.float64)
        step_pairs: dict[tuple[int, int], _ContactValue] = {}
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            body1 = self._contact_body(contact, 0)
            body2 = self._contact_body(contact, 1)
            body1_is_robot = body1 in self.robot_body_ids
            body2_is_robot = body2 in self.robot_body_ids
            if body1_is_robot == body2_is_robot:
                continue
            robot_body, other_body = (
                (body1, body2) if body1_is_robot else (body2, body1)
            )
            mujoco.mj_contactForce(self.model, data, contact_id, force)
            normal = abs(float(force[0]))
            tangent = float(np.linalg.norm(force[1:3]))
            key = robot_body, other_body
            step = step_pairs.setdefault(key, _ContactValue())
            step.position += normal * np.asarray(contact.pos)
            step.normal_force += normal
            step.tangent_force += tangent

        for key, step in step_pairs.items():
            value = self._pairs.setdefault(key, _ContactValue())
            value.normal_impulse += step.normal_force * self.model.opt.timestep
            value.sample_count += 1
            value.tangent_force = max(value.tangent_force, step.tangent_force)
            if step.normal_force >= value.normal_force:
                value.normal_force = step.normal_force
                if step.normal_force > 0.0:
                    value.position = step.position / step.normal_force

    def finish(self) -> ContactFrame:
        ranked = sorted(
            self._pairs.items(),
            key=lambda item: item[1].normal_impulse,
            reverse=True,
        )
        frame = self._empty_frame()
        for index, ((robot_body, other_body), value) in enumerate(
            ranked[:MAX_CONTACTS]
        ):
            frame.robot_body_id[index] = robot_body
            frame.other_body_id[index] = other_body
            frame.position[index] = value.position
            frame.normal_force[index] = value.normal_force
            frame.tangent_force[index] = value.tangent_force
            frame.normal_impulse[index] = value.normal_impulse
            frame.sample_count[index] = value.sample_count
        frame = replace(frame, count=len(ranked))
        self.last_frame = frame
        return frame

    def _is_descendant(self, body_id: int, root: int) -> bool:
        while body_id > 0:
            if body_id == root:
                return True
            body_id = int(self.model.body_parentid[body_id])
        return False

    def _contact_body(self, contact: mujoco.MjContact, side: int) -> int:
        geom_id = int(contact.geom[side])
        if geom_id >= 0:
            return int(self.model.geom_bodyid[geom_id])
        flex_id = int(contact.flex[side])
        return self._flex_body_ids[flex_id] if flex_id >= 0 else 0

    def _flex_root_body(self, flex_id: int) -> int:
        address = int(self.model.flex_nodeadr[flex_id])
        count = int(self.model.flex_nodenum[flex_id])
        body_ids = self.model.flex_nodebodyid[address : address + count]
        if not len(body_ids):
            address = int(self.model.flex_vertadr[flex_id])
            count = int(self.model.flex_vertnum[flex_id])
            body_ids = self.model.flex_vertbodyid[address : address + count]
        body_ids = [int(body_id) for body_id in body_ids if body_id > 0]
        if not body_ids:
            return 0

        root = body_ids[0]
        for body_id in body_ids[1:]:
            ancestors = set()
            while body_id > 0:
                ancestors.add(body_id)
                body_id = int(self.model.body_parentid[body_id])
            while root not in ancestors and root > 0:
                root = int(self.model.body_parentid[root])
        return root

    @staticmethod
    def _empty_frame() -> ContactFrame:
        return ContactFrame(
            robot_body_id=np.full(MAX_CONTACTS, -1, dtype=np.int32),
            other_body_id=np.full(MAX_CONTACTS, -1, dtype=np.int32),
            position=np.zeros((MAX_CONTACTS, 3), dtype=np.float32),
            normal_force=np.zeros(MAX_CONTACTS, dtype=np.float32),
            tangent_force=np.zeros(MAX_CONTACTS, dtype=np.float32),
            normal_impulse=np.zeros(MAX_CONTACTS, dtype=np.float32),
            sample_count=np.zeros(MAX_CONTACTS, dtype=np.int32),
            count=0,
        )
