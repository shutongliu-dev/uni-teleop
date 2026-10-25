"""Prepare 15 independent right-hand OakInk2 manipulation references.

Run with Python containing torch (for the official pickle), numpy and requests.
Runtime consumers need only numpy, and never unpickle source downloads.
Raw annotations are verified against the pinned official LFS SHA-256 catalog.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import pickle
import shutil
import time
from urllib.parse import quote

import numpy as np

from bench.paths import DATA_ROOT, OUTPUT_ROOT, load_json

REPO = "kelvin34501/OakInk-v2"
REVISION = "21705616140d726607027e70d58b7837f442ffd8"
# Fifteen different objects; one complete official primitive per object. Parts
# requiring a hinge/assembly (e.g. stapler lever) deliberately are not selected.
OBJECT_IDS = [
    "O02@0025@00001", "O02@0092@00001", "O02@0090@00001",
    "O02@0206@00002", "O02@0011@00003", "O02@0201@00001",
    "O02@0030@00002", "O02@0031@00002", "S10020", "O02@0011@00002",
    "O02@0020@00001", "C51001", "C11001", "C43001", "O02@0015@00017",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def select(source: Path) -> list[dict]:
    catalog = {Path(x["path"]).stem: x for x in load_json(source / "source/anno_preview_files.json")}
    descriptions = load_json(source / "data/object_raw/obj_desc.json")
    candidates = {}
    for path in sorted((source / "data/program/program_info").glob("*.json")):
        if path.stem not in catalog:
            continue
        program = load_json(path)
        all_ranges = [ast.literal_eval(key) for key in program]
        for key, annotation in program.items():
            left, right = ast.literal_eval(key)
            if left is not None or right is None or annotation["primitive"] != "rearrange":
                continue
            if len(annotation["obj_list"]) != 1:
                continue
            object_id = annotation["obj_list"][0]
            if object_id not in OBJECT_IDS:
                continue
            # A separately annotated left-hand action may overlap this right
            # primitive, even when its own interval tuple says left=None.
            if any(lr[0] and lr[0][0] < right[1] and lr[0][1] > right[0] for lr in all_ranges):
                continue
            row = {"sequence": path.stem, "interval_key": key,
                   "frame_range": list(right), "object_id": object_id,
                   "object_name": descriptions[object_id]["obj_name"],
                   "annotation": annotation, "download": catalog[path.stem]}
            old = candidates.get(object_id)
            if old is None or (row["download"]["size"], row["sequence"], key) < (
                    old["download"]["size"], old["sequence"], old["interval_key"]):
                candidates[object_id] = row
    missing = set(OBJECT_IDS) - set(candidates)
    if missing:
        raise ValueError(f"Required right-hand primitives not found: {missing}")
    rows = []
    for index, object_id in enumerate(OBJECT_IDS, 1):
        row = candidates[object_id]
        name = "_".join(row["object_name"].lower().split())
        row["clip_id"] = f"oi2_{index:02d}_{name}_rearrange"
        rows.append(row)
    return rows


def acquire(row: dict, source: Path, output: Path) -> dict:
    import requests
    item = row["download"]
    expected = item["lfs"]["oid"]
    existing = source / "data" / item["path"]
    target = output / "source" / item["path"]
    if existing.is_file() and sha256(existing) == expected:
        target = existing
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{quote(item['path'], safe='/')}"
    if not (target.is_file() and target.stat().st_size == item["size"] and sha256(target) == expected):
        partial = target.with_suffix(".pkl.part")
        for attempt in range(4):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                if offset > item["size"]:
                    raise ValueError("Partial file larger than pinned source")
                if offset < item["size"]:
                    with requests.get(url, headers={"Range": f"bytes={offset}-"} if offset else {},
                                      stream=True, timeout=(30, 120)) as response:
                        response.raise_for_status()
                        if offset and response.status_code != 206:
                            offset = 0
                        if response.status_code == 206 and not response.headers.get(
                                "Content-Range", "").startswith(f"bytes {offset}-"):
                            raise ValueError("Unexpected Content-Range")
                        with partial.open("ab" if offset else "wb") as handle:
                            for block in response.iter_content(1024**2):
                                handle.write(block)
                if partial.stat().st_size != item["size"] or sha256(partial) != expected:
                    partial.rename(partial.with_suffix(f".invalid-{time.time_ns()}"))
                    raise ValueError("Pinned source size/SHA-256 mismatch")
                partial.replace(target)
                break
            except (requests.RequestException, ValueError):
                if attempt == 3:
                    raise
                time.sleep(2**attempt)
    print(f"VERIFIED {row['clip_id']} {item['size']} bytes", flush=True)
    return {"path": str(target), "url": url, "sha256": expected,
            "size_bytes": item["size"], "verified": True}


def array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def read_obj_geometry(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read OBJ geometry without changing axes, units or topology."""
    vertices, triangles = [], []
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append([float(v) for v in fields[1:4]])
        elif fields[0] == "f":
            indices = [int(v.split("/")[0]) for v in fields[1:]]
            indices = [v - 1 if v > 0 else len(vertices) + v for v in indices]
            triangles.extend([indices[0], indices[j], indices[j+1]] for j in range(1, len(indices)-1))
    verts, faces = np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int64)
    if len(verts) == 0 or len(faces) == 0 or faces.min() < 0 or faces.max() >= len(verts):
        raise ValueError(f"Invalid OBJ geometry {path}")
    return verts, faces


def export(row: dict, downloaded: dict, source: Path, output: Path) -> dict:
    # This is the one trust boundary. Only SHA-256 checked official source is
    # decoded; the produced archive is primitive numeric arrays with no pickle.
    if sha256(Path(downloaded["path"])) != downloaded["sha256"]:
        raise ValueError("Source changed since download verification")
    with Path(downloaded["path"]).open("rb") as stream:
        annotation = pickle.load(stream)
    begin, end = row["frame_range"]
    all_ids = np.asarray(annotation["mocap_frame_id_list"], dtype=np.int64)
    ids = all_ids[(all_ids >= begin) & (all_ids < end)]
    if not np.array_equal(ids, np.arange(begin, end)):
        raise ValueError(f"Missing mocap frames in {row['clip_id']}")
    objects = list(annotation["obj_list"])
    target_index = objects.index(row["object_id"])
    arrays = {
        "source_frame_ids": ids,
        "source_time_s": ids.astype(np.float64) / 120.0,
        "time_s": (ids - begin).astype(np.float64) / 120.0,
        "object_ids": np.asarray(objects, dtype=np.str_),
        "object_T_world": np.stack([np.stack([
            array(annotation["obj_transf"][obj][int(fid)]) for obj in objects]) for fid in ids]),
    }
    for side in ("rh", "lh"):
        for src, dst in (("pose_coeffs", "mano_pose_wxyz"), ("tsl", "mano_tsl"), ("betas", "mano_betas")):
            arrays[f"{side}_{dst}"] = np.stack([
                array(annotation["raw_mano"][int(fid)][f"{side}__{src}"])[0] for fid in ids])
    for key, value in arrays.items():
        if value.dtype.hasobject or (value.dtype.kind == "f" and not np.isfinite(value).all()):
            raise ValueError(f"Invalid numeric array {key}")
    poses = arrays["object_T_world"]
    if not np.allclose(poses[:, :, 3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("Invalid object homogeneous pose")
    if not np.allclose(np.linalg.det(poses[:, :, :3, :3]), 1, atol=1e-3):
        raise ValueError("Invalid object SO(3)")
    out = output / "clips" / row["clip_id"]
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "reference.npz", **arrays)
    mesh_candidates = list((source / "data/object_repair/align_ds" / row["object_id"]).glob("*.obj"))
    if len(mesh_candidates) != 1:
        raise ValueError(f"Expected repaired mesh for {row['object_id']}")
    mesh_path = mesh_candidates[0]
    vertices, faces = read_obj_geometry(mesh_path)
    bounds = np.stack([vertices.min(axis=0), vertices.max(axis=0)])
    triangles = vertices[faces]
    area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)
    centroid = np.average(triangles.mean(axis=1), axis=0, weights=area)
    edges = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    watertight = bool(np.all(counts == 2))
    target_poses = poses[:, target_index]
    position = target_poses[:, :3, 3]
    wrist_proxy = arrays["lh_mano_tsl"]
    object_center = (target_poses[:, :3, :3] @ centroid[:, None]).squeeze(-1) + position
    left_distance = np.linalg.norm(wrist_proxy - object_center, axis=1)
    initial_world_vertices = vertices @ target_poses[0, :3, :3].T + position[0]
    description_path = source / "data/program/desc_info" / f"{row['sequence']}.json"
    description = load_json(description_path).get(row["interval_key"], {})
    initial_condition_path = source / "data/program/initial_condition_info" / f"{row['sequence']}.json"
    metadata = {
        "schema_version": "oakink2_right_hand_clip_v1",
        "clip_id": row["clip_id"], "dataset": "OakInk2", "repository": REPO,
        "revision": REVISION, "license": "CC-BY-SA-4.0",
        "sequence": row["sequence"], "source_annotation": downloaded,
        "source_primitive_interval_key": row["interval_key"],
        "source_frame_range": {"start": begin, "end_exclusive": end},
        "mocap_fps": 120, "duration_s": (end - begin) / 120,
        "frame_count": len(ids), "reference_path": "reference.npz",
        "reference_sha256": sha256(out / "reference.npz"),
        "annotation": row["annotation"], "description": description,
        "handedness": "right", "left_primitive_in_clip": False,
        "overlapping_left_primitive": False,
        "hand_annotation": {"representation": "MANO local joint quaternions plus global rotation, translation and shape",
            "quaternion_order": "wxyz", "pose_shape": [16, 4], "mano_model_version": "1.2",
            "decoded_joints_available": False, "hand_mesh_available": False,
            "contact_labels_available": False,
            "translation_is_wrist_joint": True,
            "translation_note": "The official preview uses ManoLayer(center_idx=0), whose output joints are centered on wrist joint 0, then adds tsl; therefore tsl equals the source wrist location.",
            "translation_evidence": [
                "https://github.com/oakink/OakInk2/blob/502a02809b50b7f5d0037f92a1ccd2a479051ad1/src/oakink2_preview/launch/viz/seg_3d.py#L116-L123",
                "https://github.com/oakink/OakInk2/blob/502a02809b50b7f5d0037f92a1ccd2a479051ad1/src/oakink2_preview/launch/viz/seg_3d.py#L192-L195",
                "https://github.com/lixiny/manotorch/blob/a2a70c591f91551078b7bb2af9b5d9f275b626e0/manotorch/manolayer.py#L212-L221"]},
        "coordinate_convention": {
            "object_T_world": "Column-vector object-local to original capture-world 4x4 transform.",
            "source_world_up_axis": "not explicitly documented in source toolkit; must calibrate before simulation",
            "source_length_unit": "meter, inferred from official preview combining unscaled object meshes and MANO joints whose decoder documents meters",
            "assumed_m_per_source_unit": 1.0,
            "unit_status": "inferred_from_official_decoder_consistency; not independently physically calibrated",
            "unit_evidence": [
                "https://github.com/lixiny/manotorch/blob/a2a70c591f91551078b7bb2af9b5d9f275b626e0/README.md#L99-L105",
                "https://github.com/oakink/OakInk2/blob/502a02809b50b7f5d0037f92a1ccd2a479051ad1/src/oakink2_preview/launch/viz/seg_3d.py#L192-L199"],
            "applied_transform": "none; all exported annotations preserve source coordinates"},
        "target_object": {"object_id": row["object_id"], "name": row["object_name"],
            "index": target_index, "mesh_path": str(mesh_path), "mesh_sha256": sha256(mesh_path),
            "mesh_variant": "official object_repair", "mesh_scale": 1.0,
            "mesh_bounds_source": bounds.tolist(), "mesh_extents_source": (bounds[1]-bounds[0]).tolist(),
            "watertight_by_edge_count": watertight, "vertex_count": len(vertices),
            "face_count": len(faces),
            "initial_T_world": target_poses[0].tolist(), "goal_T_world": target_poses[-1].tolist()},
        "source_motion_summary": {
            "target_translation_displacement_source": float(np.linalg.norm(position[-1] - position[0])),
            "target_translation_path_length_source": float(np.linalg.norm(np.diff(position, axis=0), axis=1).sum()),
            "target_translation_axis_range_source": np.ptp(position, axis=0).tolist(),
            "minimum_left_mano_translation_to_object_centroid_source": float(left_distance.min()),
            "left_wrist_axis_range_source": np.ptp(wrist_proxy, axis=0).tolist(),
            "right_wrist_axis_range_source": np.ptp(arrays["rh_mano_tsl"], axis=0).tolist(),
            "max_right_mano_quaternion_norm_error": float(np.abs(np.linalg.norm(arrays["rh_mano_pose_wxyz"], axis=-1)-1).max()),
            "initial_object_world_bounds_source": [initial_world_vertices.min(axis=0).tolist(), initial_world_vertices.max(axis=0).tolist()]},
        "scene_object_ids": objects,
        "source_initial_conditions": load_json(initial_condition_path) if initial_condition_path.exists() else None,
        "support": {"source_left_hand_required_by_primitive": False,
            "source_table_geometry_available": False, "source_object_contact_annotation_available": False,
            "simulation_support": "Must be fixed explicitly in task initialization; no hidden left-hand controller."},
        "reference_use": "Read-only motion/evaluation reference. Object poses may initialize a reset once; simulator steps must never overwrite them from this trajectory.",
        "limitations": [
            "No RGB was downloaded for this added sequence; selection is based on official task labels and 3D annotations.",
            "No overlapping left-hand primitive does not prove absence of unlabelled left-hand contact.",
            "Left translation distance is a coarse audit proxy, not a hand-surface distance or contact label.",
            "Mass, friction, compliance and table geometry are not annotated and must be fixed benchmark parameters.",
            "This v1 collection covers 15 different-object rearrangement primitives; it is not a general HOI task taxonomy.",
            "OakInk2 provides rigid object pose references; flexible-body vertex/deformation ground truth is not present.",
        ],
    }
    save_json(out / "metadata.json", metadata)
    print(f"EXPORTED {row['clip_id']}: {len(ids)} frames, displacement={metadata['source_motion_summary']['target_translation_displacement_source']:.3f}", flush=True)
    return {"schema_version": "hoi_bench_clip_entry_v1", "clip_id": row["clip_id"], "metadata_path": str(out / "metadata.json"),
            "metadata_sha256": sha256(out / "metadata.json"), "reference_path": str(out / "reference.npz"),
            "reference_sha256": metadata["reference_sha256"], "object_id": row["object_id"],
            "target_object_id": row["object_id"],
            "source": {"dataset": "OakInk2", "repository": REPO, "revision": REVISION,
                       "sequence": row["sequence"], "frame_range": [begin, end], "annotation": downloaded},
            "object_assets": [{"object_id": row["object_id"], "mesh_path": str(mesh_path),
                               "mesh_sha256": sha256(mesh_path), "variant": "object_repair"}],
            "initialization_version": None, "retarget_version": None, "evaluator_version": None,
            "object_name": row["object_name"], "frame_count": len(ids), "duration_s": metadata["duration_s"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(f"{DATA_ROOT}/datasets/oakink2_subset"))
    parser.add_argument("--output", type=Path, default=Path(f"{OUTPUT_ROOT}/hoi_bench/v1"))
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--export-downloaded", action="store_true",
                        help="Export already verified downloads to partial_manifest.json while downloading continues")
    args = parser.parse_args()
    rows = select(args.source)
    selection = {"schema_version": "hoi_bench_selection_v1", "repository": REPO,
                 "revision": REVISION, "clip_count": len(rows),
                 "total_annotation_download_bytes": sum(row["download"]["size"] for row in rows),
                 "selection_policy": "one full right-only rearrange primitive per distinct object; no overlapping left primitives; minimum source-annotation bytes then lexical sequence tie break",
                 "clips": rows}
    save_json(args.output / "selection.json", selection)
    if args.select_only:
        print(json.dumps(selection, indent=2)); return
    for name in ("DATASET_CARD.md", "TOOLKIT_README.md", "toolkit_source.json", "toolkit_meta.py"):
        shutil.copyfile(args.source / "source" / name, args.output / "source" / name)
    if args.export_downloaded:
        downloads = load_json(args.output / "source/download_manifest.json")
        clips = [export(row, downloads[row["clip_id"]], args.source, args.output)
                 for row in rows if row["clip_id"] in downloads]
        save_json(args.output / "partial_manifest.json", {"schema_version": "hoi_bench_dataset_v1",
                   "complete": False, "clip_count": len(clips), "clips": clips})
        if clips:
            save_json(args.output / "first_clip.json", clips[0])
        return
    downloads = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(acquire, row, args.source, args.output): row for row in rows}
        for job in as_completed(jobs):
            row = jobs[job]
            downloads[row["clip_id"]] = job.result()
            save_json(args.output / "source/download_manifest.json", downloads)
    clips = [export(row, downloads[row["clip_id"]], args.source, args.output) for row in rows]
    save_json(args.output / "manifest.json", {
        "schema_version": "hoi_bench_dataset_v1", "dataset": "OakInk2",
        "repository": REPO, "revision": REVISION, "license": "CC-BY-SA-4.0",
        "clip_count": len(clips), "handedness": "right", "clips": clips,
        "source_catalog": str(args.source / "source/anno_preview_files.json"),
        "source_toolkit_revision": "502a02809b50b7f5d0037f92a1ccd2a479051ad1",
        "generator": "bench.data_tools.prepare_oakink_bench", "complete": True,
        "physics_ready": False,
        "physics_readiness_note": "References are complete; simulation normalization, retarget, collision assets, initialization and evaluator are separate fixed benchmark artifacts."})
    save_json(args.output / "first_clip.json", clips[0])
    partial_manifest = args.output / "partial_manifest.json"
    partial_manifest.unlink(missing_ok=True)
    partial_manifest.symlink_to("manifest.json")


if __name__ == "__main__":
    main()
