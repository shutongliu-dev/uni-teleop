"""Append-only typed episode records and terminal artifact hashes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .schemas import (ACTION_ADAPTER, ActionEvent, AgentObservation, EpisodeResult,
                      EpisodeSpec, RobotJointCommand)


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class EpisodeRecorder:
    def __init__(self, directory: str | Path, spec: EpisodeSpec):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.spec = spec.model_copy(deep=True)
        self.closed = False
        self._sequence = 0
        (self.directory / "spec.json").write_text(_json(spec.model_dump(mode="json")) + "\n")

    def _append(self, kind: str, payload: dict) -> None:
        if self.closed:
            raise RuntimeError("episode recording is finalized")
        row = {"protocol": "agent_bench_record_v1", "sequence": self._sequence,
               "kind": kind, "payload": payload}
        with (self.directory / "events.jsonl").open("a") as file:
            file.write(_json(row) + "\n")
            file.flush()
        self._sequence += 1

    def observation(self, value: AgentObservation) -> None:
        if value.episode_id != self.spec.episode_id or value.profile != self.spec.observation_profile:
            raise ValueError("observation identity/profile does not match episode")
        self._append("agent_observation", value.model_dump(mode="json"))

    def action(self, payload: dict) -> None:
        value = ACTION_ADAPTER.validate_python(payload)
        self._append("agent_action", value.model_dump(mode="json"))

    def command(self, value: RobotJointCommand) -> None:
        self._append("robot_joint_command", value.model_dump(mode="json"))

    def lifecycle(self, value: ActionEvent) -> None:
        self._append("action_event", value.model_dump(mode="json"))

    def function_call(self, value: dict) -> None:
        self._append("bench_function", value)

    def finish(self, result: EpisodeResult) -> dict:
        if self.closed:
            raise RuntimeError("episode recording is already finalized")
        if result.spec != self.spec:
            raise ValueError("result spec differs from pinned episode spec")
        path = self.directory / "result.json"
        path.write_text(_json(result.model_dump(mode="json")) + "\n")
        hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(self.directory.iterdir()) if p.is_file()}
        manifest = dict(protocol="agent_bench_artifacts_v1", episode_id=self.spec.episode_id,
                        event_count=self._sequence, files_sha256=hashes)
        (self.directory / "artifacts.json").write_text(_json(manifest) + "\n")
        self.closed = True
        return manifest
