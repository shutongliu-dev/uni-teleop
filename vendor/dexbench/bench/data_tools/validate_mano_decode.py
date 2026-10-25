"""Validate decoded geometry and compare the adapter to original MANO loading."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np

from .decode_mano import JOINT_ORDER, load_layer, numeric_ready_arguments
from .prepare_oakink_bench import save_json, sha256
from bench.paths import DATA_ROOT, OUTPUT_ROOT, load_json


def validate(output: Path, model: Path, decoder: Path, dependencies: Path, official_root: Path):
    import torch
    manifest_path = output / "manifest.json"
    manifest = load_json(manifest_path)
    source_path = Path(manifest["source_manifest_path"])
    if sha256(source_path) != manifest["source_manifest_sha256"]:
        raise ValueError("Original source manifest changed")
    if manifest["joint_order"] != JOINT_ORDER or not manifest["complete"]:
        raise ValueError("Incomplete decoded collection or incorrect joint order")
    if sha256(model) != manifest["model_and_decoder"]["model_sha256"]:
        raise ValueError("MANO model hash mismatch")
    layer, _ = load_layer(model, decoder, dependencies)
    # Run the original ready_arguments function verbatim. The released module's
    # unrelated __main__ has Python 2 print syntax, so compile only the exact
    # prefix containing this function; pass an already decoded dict to avoid its
    # Python 2 text-mode pickle open. This comparison does not alter source files.
    official_file = official_root / "webuser/smpl_handpca_wrapper_HAND_only.py"
    prefix = official_file.read_text().split("\ndef load_model(", 1)[0]
    sys.path.insert(0, str(official_root))
    sys.modules["cPickle"] = pickle
    namespace = {}
    exec(compile(prefix, str(official_file), "exec"), namespace)
    with model.open("rb") as stream:
        original_model = pickle.load(stream, encoding="latin1")
    original = namespace["ready_arguments"](original_model)
    adapted = numeric_ready_arguments(model)
    buffer_checks = {}
    for key in ("betas", "shapedirs", "posedirs", "v_template", "weights", "f", "kintree_table", "hands_components", "J_regressor"):
        def values(value):
            if hasattr(value, "toarray"):
                return value.toarray()
            return np.asarray(value.r if hasattr(value, "r") else value)
        a, b = values(original[key]), values(adapted[key])
        if not np.array_equal(a, b):
            raise ValueError(f"Adapter differs from original MANO ready_arguments buffer: {key}")
        buffer_checks[key] = {"exact_match": True, "shape": list(a.shape)}
    max_error, max_wrist_error, frame_total, samples = 0., 0., 0, 0
    reports = []
    for entry in manifest["clips"]:
        if sha256(Path(entry["decoded_path"])) != entry["decoded_sha256"]:
            raise ValueError("Decoded archive changed")
        meta_path = Path(entry["metadata_path"])
        if sha256(meta_path) != entry["metadata_sha256"]:
            raise ValueError("Decoded metadata changed")
        metadata = load_json(meta_path)
        if sha256(Path(metadata["source_reference_path"])) != metadata["source_reference_sha256"]:
            raise ValueError("Original numeric reference changed")
        with np.load(entry["decoded_path"], allow_pickle=False) as data:
            n = entry["frame_count"]
            frame_total += n
            if data["rh_joints_world_m"].shape != (n, 21, 3) or data["joint_order"].tolist() != JOINT_ORDER:
                raise ValueError("Incorrect 21-joint archive layout")
            if data["rh_vertices_world_m"].shape != (n, 778, 3):
                raise ValueError("Incorrect 778-vertex archive layout")
            if data["mano_faces"].shape[1] != 3 or data["mano_faces"].min() < 0 or data["mano_faces"].max() > 777:
                raise ValueError("Invalid MANO triangle topology")
            if not np.array_equal(data["mano_faces"], layer.th_faces.numpy()):
                raise ValueError("Faces differ from licensed model")
            for key in data.files:
                if data[key].dtype.hasobject or (data[key].dtype.kind == "f" and not np.isfinite(data[key]).all()):
                    raise ValueError("Nonfinite or pickle-dependent output")
            wrist_error = float(np.max(np.abs(data["rh_joints_world_m"][:, 0] - data["rh_mano_tsl"])))
            max_wrist_error = max(max_wrist_error, wrist_error)
            if wrist_error > 1e-7:
                raise ValueError("Wrist/translation mismatch")
            indices = sorted(set([0, n//2, n-1]))
            local_max = 0.
            for index in indices:
                with torch.no_grad():
                    pose = torch.tensor(data["rh_mano_pose_wxyz"][index:index+1], dtype=torch.float32)
                    betas = torch.tensor(data["rh_mano_betas"][index:index+1], dtype=torch.float32)
                    result = layer(pose_coeffs=pose, betas=betas)
                translation = data["rh_mano_tsl"][index:index+1, None]
                error = max(float(np.max(np.abs(result.joints.numpy() + translation - data["rh_joints_world_m"][index:index+1]))),
                            float(np.max(np.abs(result.verts.numpy() + translation - data["rh_vertices_world_m"][index:index+1]))))
                if error > 1e-6:
                    raise ValueError("Single-frame decode differs from batched export")
                local_max = max(local_max, error)
                samples += 1
            max_error = max(max_error, local_max)
            reports.append({"clip_id": entry["clip_id"], "frames": n, "sample_frames_redecoded": indices,
                            "max_redecode_error_m": local_max, "max_wrist_error_m": wrist_error})
    if frame_total != manifest["total_frames"] or len(reports) != manifest["clip_count"]:
        raise ValueError("Collection totals mismatch")
    return {"schema_version": "mano_decode_audit_v1", "passed": True,
        "decoder_manifest_path": str(manifest_path), "decoder_manifest_sha256": sha256(manifest_path),
        "model_sha256": sha256(model), "right_hand_only": True,
        "clip_count": len(reports), "total_frames": frame_total,
        "per_frame_joint_count": 21, "per_frame_vertex_count": 778,
        "source_manifest_unchanged": True, "original_mano_loader_buffers": buffer_checks,
        "original_loader_path": str(official_file), "original_loader_sha256": sha256(official_file),
        "original_loader_validation_note": "Unmodified ready_arguments function compiled without unrelated Python 2 __main__; cPickle alias and dict input supply Python 3 compatibility.",
        "single_frame_redecode_count": samples, "max_redecode_error_m": max_error,
        "max_wrist_error_m": max_wrist_error, "joint_order": JOINT_ORDER, "clips": reports,
        "scope": "Exact licensed model buffers, pinned upstream decoder, topology/units/root convention and numeric export consistency; not physical contact or robot-retarget validation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(f"{OUTPUT_ROOT}/hoi_bench/decoded_mano_v1"))
    parser.add_argument("--model", type=Path, default=Path(f"{DATA_ROOT}/assets/body_models/mano/v1.2/models/MANO_RIGHT.pkl"))
    parser.add_argument("--decoder", type=Path, default=Path(f"{DATA_ROOT}/assets/body_models/mano/decoders/manotorch"))
    parser.add_argument("--dependencies", type=Path, default=Path(f"{DATA_ROOT}/assets/body_models/mano/decoders/python-deps/chumpy-0.70"))
    parser.add_argument("--official-root", type=Path, default=Path(f"{DATA_ROOT}/assets/body_models/mano/v1.2/upstream/mano_v1_2"))
    args = parser.parse_args()
    result = validate(args.output, args.model, args.decoder, args.dependencies, args.official_root)
    save_json(args.output / "decode_audit.json", result)
    print(json.dumps({key: value for key, value in result.items() if key not in ("clips", "original_mano_loader_buffers", "joint_order")}, indent=2))


if __name__ == "__main__":
    main()
