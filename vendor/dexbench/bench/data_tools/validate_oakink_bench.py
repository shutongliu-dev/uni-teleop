"""Audit prepared references against the pinned source labels and file hashes."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import numpy as np

from .prepare_oakink_bench import sha256, save_json
from bench.paths import DATA_ROOT, load_json


def validate(root: Path, source: Path) -> dict:
    manifest = load_json(root / "manifest.json")
    if manifest["clip_count"] != 15 or not manifest["complete"]:
        raise ValueError("Expected the complete fixed collection of 15 clips")
    clip_ids, target_ids, report = set(), set(), []
    for entry in manifest["clips"]:
        clip_id = entry["clip_id"]
        if clip_id in clip_ids or entry["target_object_id"] in target_ids:
            raise ValueError("Repeated clip or target object")
        clip_ids.add(clip_id); target_ids.add(entry["target_object_id"])
        meta_path = Path(entry["metadata_path"])
        if sha256(meta_path) != entry["metadata_sha256"]:
            raise ValueError(f"Changed metadata {clip_id}")
        meta = load_json(meta_path)
        ref_path = Path(entry["reference_path"])
        if sha256(ref_path) != entry["reference_sha256"]:
            raise ValueError(f"Changed reference {clip_id}")
        raw = meta["source_annotation"]
        if sha256(Path(raw["path"])) != raw["sha256"]:
            raise ValueError(f"Changed source annotation {clip_id}")
        programs = load_json(source / "data/program/program_info" / (meta["sequence"] + ".json"))
        key = meta["source_primitive_interval_key"]
        left, (begin, end) = ast.literal_eval(key)
        if left is not None or programs[key] != meta["annotation"]:
            raise ValueError(f"Not an unchanged right-only primitive: {clip_id}")
        for other_key in programs:
            other_left, _ = ast.literal_eval(other_key)
            if other_left and other_left[0] < end and other_left[1] > begin:
                raise ValueError(f"Overlapping left-hand primitive: {clip_id}")
        with np.load(ref_path, allow_pickle=False) as ref:
            expected_frames = np.arange(begin, end)
            if not np.array_equal(ref["source_frame_ids"], expected_frames):
                raise ValueError("Reference has gaps, reordered frames or cropped primitive")
            if not np.allclose(ref["time_s"], (expected_frames - begin) / 120):
                raise ValueError("Incorrect mocap clock")
            if not np.allclose(ref["source_time_s"], expected_frames / 120):
                raise ValueError("Incorrect original timestamps")
            for side in ("rh", "lh"):
                if ref[f"{side}_mano_pose_wxyz"].shape != (len(expected_frames), 16, 4):
                    raise ValueError("Incorrect 3D MANO pose layout")
                if not np.allclose(np.linalg.norm(ref[f"{side}_mano_pose_wxyz"], axis=-1), 1, atol=1e-5):
                    raise ValueError("Non-unit MANO quaternion")
                if ref[f"{side}_mano_tsl"].shape != (len(expected_frames), 3):
                    raise ValueError("Incorrect 3D wrist layout")
                if ref[f"{side}_mano_betas"].shape != (len(expected_frames), 10):
                    raise ValueError("Incorrect MANO shape layout")
            poses = ref["object_T_world"]
            if poses.shape != (len(expected_frames), len(ref["object_ids"]), 4, 4):
                raise ValueError("Incorrect object pose layout")
            rot = poses[..., :3, :3]
            if not np.allclose(np.swapaxes(rot, -1, -2) @ rot, np.eye(3), atol=1e-5):
                raise ValueError("Non-orthogonal object rotation")
            if not np.allclose(np.linalg.det(rot), 1, atol=1e-5):
                raise ValueError("Improper object rotation")
            target_index = meta["target_object"]["index"]
            if ref["object_ids"][target_index] != entry["target_object_id"]:
                raise ValueError("Target object ordering mismatch")
            if not np.array_equal(poses[0, target_index], meta["target_object"]["initial_T_world"]):
                raise ValueError("Initial pose differs from source reference")
            if not np.array_equal(poses[-1, target_index], meta["target_object"]["goal_T_world"]):
                raise ValueError("Goal pose differs from source reference")
            if not all(np.isfinite(ref[k]).all() for k in ref.files if ref[k].dtype.kind in "fc"):
                raise ValueError("Nonfinite reference values")
        for asset in entry["object_assets"]:
            if sha256(Path(asset["mesh_path"])) != asset["mesh_sha256"]:
                raise ValueError(f"Changed object asset {asset['object_id']}")
        if not meta["hand_annotation"]["translation_is_wrist_joint"]:
            raise ValueError("Wrist convention disagrees with official center_idx=0 decoder")
        motion = meta["source_motion_summary"]
        report.append({"clip_id": clip_id, "frames": end-begin,
                       "target_displacement_source": motion["target_translation_displacement_source"],
                       "min_left_wrist_to_target_centroid_source": motion["minimum_left_mano_translation_to_object_centroid_source"],
                       "initial_source_y_min": motion["initial_object_world_bounds_source"][0][1],
                       "asset_extents_source": meta["target_object"]["mesh_extents_source"]})
    return {"schema_version": "hoi_reference_audit_v1", "passed": True,
            "clip_count": len(report), "distinct_target_objects": len(target_ids),
            "total_frames": sum(c["frames"] for c in report), "clips": report,
            "scope": "Source provenance, exact full primitive windows, 3D annotations, object meshes and hashes; does not certify physical retarget feasibility or simulator task success."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(f"{DATA_ROOT}/datasets/hoi_bench/v1"))
    parser.add_argument("--source", type=Path, default=Path(f"{DATA_ROOT}/datasets/oakink2_subset"))
    args = parser.parse_args()
    result = validate(args.root, args.source)
    save_json(args.root / "reference_audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "clips"}, indent=2))


if __name__ == "__main__":
    main()
