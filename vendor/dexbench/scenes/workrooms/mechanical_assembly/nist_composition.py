"""Independent, deduplicated NIST-derived boards in the fixed mechanical room."""
from copy import deepcopy
import json
from pathlib import Path

from bench.isaacsim.release import checked, digest
from .task_scene import empty_task_scene
from bench.paths import load_json

TASK_IDS = (
    "nist_connector_insertion", "nist_belt_drive_assembly",
    "nist_cable_routing", "nist_wire_harness",
)
TASK_SCHEMA = "mechanical-nist-task-v1"
ASSET_SCHEMA = "mechanical-nist-task-asset.v1"
ROOT_PRIM = "/World/NistTask"
PHYSICS_DT = .0005
TITLES = (
    "NIST ATB1 · 电气接头插接", "NIST ATB2 · 滑轮皮带装配",
    "NIST ATB3 · 线缆绕障布线", "NIST ATB4 · 多线束装配与取下",
)
SOURCE_URL = "https://www.nist.gov/el/intelligent-systems-division-73500/robotic-grasping-and-manipulation-assembly/assembly"


def evaluator_sources(task_id):
    if task_id not in TASK_IDS:
        raise ValueError("Unknown NIST-derived task")
    root = Path(__file__).parent
    return {"sequence_evaluator.py": root/"nist_evaluation.py",
            "rules.py": root/"nist_rules.py",
            "native_evidence.py": root/"nist_native_evidence.py"}


def load_asset_manifest(task_id, asset_manifest):
    if task_id not in TASK_IDS:
        raise ValueError("Unknown NIST-derived task")
    if isinstance(asset_manifest, dict):
        path = Path(asset_manifest["manifest_path"]).resolve()
        checked(path, asset_manifest["manifest_sha256"])
    else:
        path = Path(asset_manifest).resolve()
    manifest = load_json(path)
    if (manifest.get("schema") != ASSET_SCHEMA or manifest.get("task_id") != task_id
            or manifest.get("root_prim_path") != ROOT_PRIM
            or manifest.get("presentation") is not False):
        raise ValueError("Expected a dynamic NIST task reset asset")
    usd, meta = path.parent/"task.usda", path.parent/"metadata.json"
    checked(usd, manifest["usd_sha256"])
    checked(meta, manifest["metadata_sha256"])
    auxiliary = {str(path.parent/name): manifest[key] for name, key in (
        ("structural_audit.json", "structural_audit_sha256"),
        ("ATTRIBUTION.txt", "attribution_sha256"))}
    for filename, sha in auxiliary.items():
        checked(filename, sha)
    audit = load_json(path.parent/"structural_audit.json")
    if audit.get("passed") is not True or audit.get("errors"):
        raise ValueError("Task asset topology audit did not pass")
    for dependency, row in manifest["dependencies"].items():
        checked(dependency, row["sha256"])
    metadata = load_json(meta)
    if (metadata.get("task_id") != task_id
            or metadata.get("root_path") != ROOT_PRIM
            or metadata.get("presentation") is not False):
        raise ValueError("NIST task metadata differs from its asset identity")
    actors = metadata["actors"]
    if (not actors or len(set(actors.values())) != len(actors)
            or any(not p.startswith(ROOT_PRIM+"/") for p in actors.values())):
        raise ValueError("Distinct task-owned dynamic actors are required")
    return {"manifest_path": str(path), "manifest_sha256": digest(path),
            "usd_path": str(usd), "usd_sha256": manifest["usd_sha256"],
            "metadata_path": str(meta), "metadata_sha256": manifest["metadata_sha256"],
            "root_prim_path": ROOT_PRIM, "dependencies": deepcopy(manifest["dependencies"]),
            "auxiliary_files_sha256": auxiliary}, metadata


def bind_episode_task(task, seed, rules):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Episode seed must be an unsigned 32-bit integer")
    task = deepcopy(task)
    task["episode_seed"] = seed
    task["public_rules"] = rules.public_rules(task["id"], seed=seed)
    task["evaluation"] = rules.evaluation_configuration(
        task["id"], task["asset_metadata"], seed=seed, physics_dt_s=PHYSICS_DT,
        room_support_paths=task["room_support_paths"])
    task["public_goal"] = {"operation": task["id"], "frame_id": "world_m_z_up",
                          "description": task["public_rules"]["instruction_zh"]}
    return rules.bind_task_rules(task)


def build_task_inputs(task_id, *, asset_manifest, seed=0):
    from . import nist_rules as rules
    candidate, metadata = load_asset_manifest(task_id, asset_manifest)
    scene, initial, ready = empty_task_scene()
    description = load_json(Path(scene["workroom"]["dependency_manifest_path"]))
    scene["isaacsim"]["physics_dt_s"] = PHYSICS_DT
    scene["physics"] = {"dt_s": PHYSICS_DT, "control_dt_s": .02,
                        "gravity_m_s2": [0, 0, -9.81], "backend": "native_isaacsim_physx_gpu"}
    # A frozen physical reset is reused across episode identity seeds. The
    # legacy scenario envelope instead requires one fixed initialization seed.
    scene.pop("scenario", None)
    scene["description"] = TITLES[TASK_IDS.index(task_id)] + "；去重后的 NIST 改编任务。"
    scene["cameras"]["task"].update(eye=[1.12, -.82, 1.68], target=[.43, 0., .83],
                                    focal_length_mm=32.)
    task = {
        "schema_version": TASK_SCHEMA, "id": task_id, "task_id": task_id,
        "title_zh": TITLES[TASK_IDS.index(task_id)], "difficulty": TASK_IDS.index(task_id)+6,
        "cloth": False, "initial_state": deepcopy(initial), "asset_metadata": metadata,
        "room_support_paths": description["support_colliders"],
        "steps": [{"id": task_id, "instruction_zh": "完成公开装配目标，允许纠正和重试，松手后稳定。"}],
        "recommended_budgets": {"sim_time_s": 180., "wall_time_s": 2400., "max_actions": 9000},
        "training_policy_included": False, "physics_accepted": False,
        "task_policy_success_claimed": False,
        "source_benchmark": {"url": SOURCE_URL, "board": TASK_IDS.index(task_id)+1,
            "scope": "Deduplicated adaptation, not the complete official NIST board or test protocol."},
        "known_limitations": [
            "NIST-derived simulation adaptation; no official benchmark equivalence or certification.",
            "Materials and flexible-body parameters are declared simulation priors.",
            "No electrical continuity, connector latch strength or physical hardware calibration is claimed.",
            "Scene feasibility and full robot manipulation are separate qualifications.",
        ] + metadata.get("known_limitations", metadata.get("limitations", [])),
    }
    task = bind_episode_task(task, seed, rules)
    sources = evaluator_sources(task_id)
    scene["native_task_bundle"] = {
        "schema": "native-usd-task-bundle-v1", "usd_path": candidate["usd_path"],
        "usd_sha256": candidate["usd_sha256"], "root_prim_path": ROOT_PRIM,
        "rigid_bodies": deepcopy(metadata["actors"]), "joints": deepcopy(metadata.get("joints", {})),
        "particle_sets": {},
        "observer": {"path": "evaluator/sequence_evaluator.py",
            "sha256": digest(sources["sequence_evaluator.py"]),
            "files": {name: digest(path) for name, path in sources.items()},
            "class_name": "NativeTaskObserver", "configuration": deepcopy(task)},
    }
    scene["mechanical_nist_provenance"] = {
        "asset_candidate": candidate, "ready_design": ready,
        "parent_environment_unchanged": True, "shared_controller_unchanged": True,
        "physical_reset_seeded": False, "seed_scope": "Public rules; physical reset is frozen USD.",
        "task_object_pose_writes_after_reset": False,
    }
    return {"scene": scene, "task": task, "initial_state": initial}
