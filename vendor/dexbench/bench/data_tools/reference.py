"""Hash-verified, read-only numeric HOI references; no simulator access."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np

from bench.paths import load_json


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


class ReferenceClip:
    def __init__(self, entry: dict):
        self.entry = entry
        for key in ("reference", "metadata"):
            path = Path(entry[f"{key}_path"])
            if digest(path) != entry[f"{key}_sha256"]:
                raise ValueError(f"{key} hash mismatch: {path}")
        self.metadata = load_json(Path(entry["metadata_path"]))
        with np.load(entry["reference_path"], allow_pickle=False) as data:
            self.arrays = {k: data[k].copy() for k in data.files}
        required = {"time_s", "source_frame_ids", "rh_mano_pose_wxyz", "rh_mano_tsl",
                    "object_T_world", "object_ids"}
        if not required <= self.arrays.keys():
            raise ValueError("Missing required reference fields")
        t = self.arrays["time_s"]
        if len(t) < 2 or not np.all(np.diff(t) > 0):
            raise ValueError("Reference times must strictly increase")
        for key, value in self.arrays.items():
            if value.dtype.hasobject or (value.dtype.kind == "f" and not np.isfinite(value).all()):
                raise ValueError(f"Invalid reference array: {key}")
            value.flags.writeable = False
        self.target_index = self.arrays["object_ids"].tolist().index(entry["target_object_id"])
        self.target_poses = self.arrays["object_T_world"][:, self.target_index]
        for asset in entry["object_assets"]:
            if digest(asset["mesh_path"]) != asset["mesh_sha256"]:
                raise ValueError("Object mesh hash mismatch")

    def nearest_index(self, time_s):
        t = self.arrays["time_s"]
        i = int(np.clip(np.searchsorted(t, time_s), 0, len(t)-1))
        return i-1 if i > 0 and abs(t[i-1]-time_s) <= abs(t[i]-time_s) else i

    def hand_at(self, time_s):
        index = self.nearest_index(time_s)
        return self.arrays["rh_mano_tsl"][index], self.arrays["rh_mano_pose_wxyz"][index]


def load_entries(path):
    value = load_json(Path(path))
    if isinstance(value, list):
        return value
    if "clip_id" in value:
        return [value]
    return value["clips"]
