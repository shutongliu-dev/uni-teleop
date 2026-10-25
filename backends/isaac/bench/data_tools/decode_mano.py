"""Decode licensed MANO v1.2 with the pinned, unchanged official manotorch LBS.

Only the legacy weight-loading wrapper is adapted. Numerical skinning, shape,
pose blending and fingertip selection use upstream manotorch/manolayer.py.
Outputs preserve the original OakInk capture world and do not change physics.
"""
from __future__ import annotations

import argparse
from collections import namedtuple
import inspect
from pathlib import Path
import pickle
import sys
import types

import numpy as np

from .prepare_oakink_bench import save_json, sha256
from bench.paths import DATA_ROOT, OUTPUT_ROOT, load_json

DECODER_REVISION = "a2a70c591f91551078b7bb2af9b5d9f275b626e0"
JOINT_ORDER = ["wrist"] + [f"{finger}_{joint}" for finger in
    ("thumb", "index", "middle", "ring", "pinky") for joint in ("mcp", "pip", "dip", "tip")]
# MANO thumb joint naming is anatomical CMC/MCP/IP, although the 21-point
# convention often uses MCP/PIP/DIP for all fingers. Use the anatomical names.
JOINT_ORDER[1:5] = ["thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip"]


class NumericField:
    """The constant .r interface used by upstream to register model buffers."""
    def __init__(self, value):
        self.r = np.asarray(value.r if hasattr(value, "r") else value)


def prepare_legacy_pickle_compatibility(dependencies: Path) -> None:
    """Apply compatibility only inside this decoder process, never on disk."""
    if dependencies.is_dir():
        sys.path.insert(0, str(dependencies))
    # chumpy 0.70 imports aliases removed by numpy 1.24 and inspect.getargspec
    # removed by Python 3.11. No numeric behavior or model values are changed.
    for name, value in {"bool": bool, "int": int, "float": float, "complex": complex,
                        "object": object, "str": str, "unicode": str}.items():
        if name not in np.__dict__:
            setattr(np, name, value)
    if not hasattr(inspect, "getargspec"):
        ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")
        def getargspec(function):
            spec = inspect.getfullargspec(function)
            return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)
        inspect.getargspec = getargspec


def numeric_ready_arguments(path):
    """Read exact model buffers, avoiding unused legacy chumpy graph building.

    Upstream ManoLayer reads .r from five model buffers and the zero default
    shape; all other fields pass through untouched. Per-frame betas are supplied
    explicitly to ManoLayer, so its default betas do not affect decoded data.
    """
    with open(path, "rb") as stream:
        model = pickle.load(stream, encoding="latin1")
    required = {"shapedirs", "posedirs", "v_template", "weights", "J_regressor",
                "f", "kintree_table", "hands_components"}
    if not required <= model.keys():
        raise ValueError(f"Missing MANO model fields: {required - model.keys()}")
    result = dict(model)
    result["betas"] = NumericField(np.zeros(np.shape(model["shapedirs"])[-1], dtype=np.float64))
    for key in ("shapedirs", "posedirs", "v_template", "weights"):
        result[key] = NumericField(model[key])
    return result


def load_layer(model_path: Path, decoder_path: Path, dependencies: Path):
    import torch
    source = load_json(decoder_path / "SOURCE.json")
    if source["revision"] != DECODER_REVISION:
        raise ValueError("Unexpected manotorch decoder revision")
    # Verify every Python file in the source snapshot before importing it.
    for relative, expected in source["files"].items():
        if sha256(decoder_path / relative) != expected:
            raise ValueError(f"Changed decoder source: {relative}")
    prepare_legacy_pickle_compatibility(dependencies)
    # Upstream's only import of the legacy MANO webuser package is this buffer
    # loader. This process-local adapter preserves all weights and equations.
    for name in ("mano", "mano.webuser"):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    wrapper = types.ModuleType("mano.webuser.smpl_handpca_wrapper_HAND_only")
    wrapper.ready_arguments = numeric_ready_arguments
    sys.modules[wrapper.__name__] = wrapper
    sys.path.insert(0, str(decoder_path))
    from manotorch.manolayer import ManoLayer
    if model_path.name != "MANO_RIGHT.pkl" or model_path.parent.name != "models":
        raise ValueError("Expected <asset-root>/models/MANO_RIGHT.pkl")
    torch.set_num_threads(4)
    layer = ManoLayer(mano_assets_root=str(model_path.parent.parent), side="right",
                      center_idx=0, rot_mode="quat", use_pca=False, flat_hand_mean=True)
    return layer.to(device="cpu", dtype=torch.float32).eval(), source


def decode_clip(entry: dict, layer, output: Path, provenance: dict, with_vertices: bool, batch_size: int):
    import torch
    source_path = Path(entry["reference_path"])
    if sha256(source_path) != entry["reference_sha256"]:
        raise ValueError(f"Changed source reference {entry['clip_id']}")
    with np.load(source_path, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in (
            "source_frame_ids", "source_time_s", "time_s", "rh_mano_pose_wxyz", "rh_mano_tsl", "rh_mano_betas")}
    joints, vertices = [], []
    total = len(arrays["time_s"])
    with torch.no_grad():
        for start in range(0, total, batch_size):
            stop = min(total, start + batch_size)
            rotation = torch.as_tensor(arrays["rh_mano_pose_wxyz"][start:stop], dtype=torch.float32)
            betas = torch.as_tensor(arrays["rh_mano_betas"][start:stop], dtype=torch.float32)
            translation = torch.as_tensor(arrays["rh_mano_tsl"][start:stop], dtype=torch.float32)
            decoded = layer(pose_coeffs=rotation, betas=betas)
            if not torch.allclose(decoded.joints[:, 0], torch.zeros_like(decoded.joints[:, 0]), atol=1e-7):
                raise ValueError("Upstream output is not wrist centered")
            joints.append((decoded.joints + translation[:, None]).numpy())
            if with_vertices:
                vertices.append((decoded.verts + translation[:, None]).numpy())
    arrays["rh_joints_world_m"] = np.concatenate(joints, axis=0)
    arrays["joint_order"] = np.asarray(JOINT_ORDER, dtype=np.str_)
    if with_vertices:
        arrays["rh_vertices_world_m"] = np.concatenate(vertices, axis=0)
        arrays["mano_faces"] = layer.th_faces.cpu().numpy()
    if arrays["rh_joints_world_m"].shape != (total, 21, 3):
        raise ValueError("Unexpected official 21-joint layout")
    if not np.allclose(arrays["rh_joints_world_m"][:, 0], arrays["rh_mano_tsl"], atol=1e-7):
        raise ValueError("Decoded wrist differs from annotated root translation")
    for key, value in arrays.items():
        if value.dtype.hasobject or (value.dtype.kind == "f" and not np.isfinite(value).all()):
            raise ValueError(f"Invalid decoded array {key}")
    out = output / "clips" / entry["clip_id"]
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "hand_keypoints.npz", **arrays)
    metadata = {"schema_version": "oakink2_decoded_mano_v1", "clip_id": entry["clip_id"],
        "source_reference_path": str(source_path), "source_reference_sha256": entry["reference_sha256"],
        "source_metadata_path": entry["metadata_path"], "source_metadata_sha256": entry["metadata_sha256"],
        "decoded_path": str(out / "hand_keypoints.npz"), "decoded_sha256": sha256(out / "hand_keypoints.npz"),
        "frame_count": total, "handedness": "right", "joint_order": JOINT_ORDER,
        "coordinate_frame": "original OakInk capture world, no rotation/scale/translation conversion",
        "length_unit": "meter", "includes_vertices": with_vertices,
        "model_and_decoder": provenance,
        "checks": {"finite": True, "wrist_equals_annotation_translation": True, "shape_21_by_3": True},
        "limitations": ["Model-derived joints and optional surface are not measured contact labels.",
            "MANO shape/pose model approximation remains; decoding does not ensure robot retarget feasibility.",
            "Keep MANO model weights under their original license; do not include them in public benchmark distributions."]}
    save_json(out / "metadata.json", metadata)
    print(f"DECODED {entry['clip_id']} {total} frames", flush=True)
    return {"clip_id": entry["clip_id"], "decoded_path": metadata["decoded_path"],
            "decoded_sha256": metadata["decoded_sha256"], "metadata_path": str(out / "metadata.json"),
            "metadata_sha256": sha256(out / "metadata.json"), "frame_count": total,
            "source_reference_sha256": entry["reference_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(f"{DATA_ROOT}/datasets/hoi_bench/v1/manifest.json"))
    parser.add_argument("--model", type=Path, default=Path(f"{DATA_ROOT}/assets/body_models/mano/v1.2/models/MANO_RIGHT.pkl"))
    parser.add_argument("--decoder", type=Path, default=Path(f"{DATA_ROOT}/assets/body_models/mano/decoders/manotorch"))
    parser.add_argument("--dependencies", type=Path, default=Path(f"{DATA_ROOT}/assets/body_models/mano/decoders/python-deps/chumpy-0.70"))
    parser.add_argument("--output", type=Path, default=Path(f"{OUTPUT_ROOT}/hoi_bench/decoded_mano_v1"))
    parser.add_argument("--with-vertices", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if not args.model.is_file():
        raise FileNotFoundError(f"Licensed MANO weights are not ready: {args.model}")
    manifest = load_json(args.manifest)
    if not manifest["complete"]:
        raise ValueError("Source collection must be complete before decoding")
    layer, decoder_source = load_layer(args.model, args.decoder, args.dependencies)
    import torch
    provenance = {"model": "MANO", "model_version": "1.2", "model_path": str(args.model),
        "model_sha256": sha256(args.model), "model_license": "MANO original license stored with licensed asset",
        "decoder": "manotorch", "decoder_repository": decoder_source["repository"],
        "decoder_revision": DECODER_REVISION, "decoder_source_manifest_sha256": sha256(args.decoder / "SOURCE.json"),
        "model_loader_adapter": "bench.data_tools.decode_mano.numeric_ready_arguments",
        "adapter_sha256": sha256(Path(__file__)),
        "loader_note": "Numeric model buffers are unchanged; legacy chumpy ready_arguments graph construction is replaced only at the weight-loading boundary.",
        "settings": {"side": "right", "center_idx": 0, "rot_mode": "quat", "use_pca": False, "flat_hand_mean": True},
        "dtype": "float32", "device": "cpu", "torch_version": torch.__version__,
        "numpy_version": np.__version__, "length_unit": "meter"}
    rows = [decode_clip(entry, layer, args.output, provenance, args.with_vertices, args.batch_size)
            for entry in manifest["clips"]]
    save_json(args.output / "manifest.json", {"schema_version": "oakink2_decoded_collection_v1",
        "source_manifest_path": str(args.manifest), "source_manifest_sha256": sha256(args.manifest),
        "complete": True, "clip_count": len(rows), "total_frames": sum(r["frame_count"] for r in rows),
        "joint_order": JOINT_ORDER, "model_and_decoder": provenance, "clips": rows})


if __name__ == "__main__":
    main()
