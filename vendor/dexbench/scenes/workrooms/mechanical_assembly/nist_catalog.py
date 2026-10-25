"""Task6–9 definitions; original five task releases remain independent."""
from copy import deepcopy
from .nist_composition import TASK_IDS, TITLES
from .nist_rules import public_rules, thresholds
from .task_design import PRIVILEGED_VALIDATION
from bench.paths import DATA_ROOT

ASSET_ROOT = f"{DATA_ROOT}/assets/task_parts/mechanical_assembly"
RELEASE_ROOT = f"{DATA_ROOT}/datasets/fr3_sharpa_bench/isaacsim/releases/mechanical_assembly"


def task_definition(task_id, *, manifest_sha256=None, asset_version="v1",
                    release_version="v1", evidence=(), native_preflight=False,
                    object_interactions=False):
    index = TASK_IDS.index(task_id)
    number = index+6
    implemented = manifest_sha256 is not None
    rules = public_rules(task_id)
    result = {
        "schema": "workroom_task_definition.v1", "family": "mechanical_assembly",
        "task_id": task_id, "task_number": number, "difficulty": number,
        "difficulty_calibrated": False, "title": f"Task{number} · {TITLES[index]}",
        "status": "implemented" if implemented else "planned",
        "entrypoint": f"scenes.workrooms.mechanical_assembly.tasks.{task_id}.task",
        "release_schema": "mechanical_nist_task_release.v1",
        "release_schema_version": "isaacsim-mechanical-nist-release-v1",
        "objects": {"intended_roles": (
            ["connector_plugs", "matching_sockets"],
            ["pulleys", "passive_link_belt", "tensioner"],
            ["cable", "routing_guides", "end_connector"],
            ["multiple_wire_leads", "paired_terminals", "routing_board", "collection_tray"],
        )[index], "asset_root": f"{ASSET_ROOT}/{task_id}/{asset_version}",
            "asset_bindings": "Source-pinned task USD and physical metadata." if implemented else None},
        "layout": {"frame": "world", "workspace": "mechanical_table_near_side",
            "parent_scene": "mechanical_assembly/scene.py", "same_environment_for_all_tasks": True,
            "initialization": "Frozen supply reset and original ready robot." if implemented else None},
        "robot_requirements": {"arm": "fr3", "hand_id": "sharpa_w01_right",
            "assembly_version": "isaacsim_6_1_v2", "articulations": 1,
            "layout_id": "mechanical_single_arm_v1", "base_position_m": [-.23, 0, .78]},
        "rules": {"intent": rules["instruction_zh"],
            "implementation": f"scenes.workrooms.mechanical_assembly.tasks.{task_id}.rules",
            "contract": rules if implemented else None},
        "evaluator": {"implementation": "scenes.workrooms.mechanical_assembly.nist_evaluation.SequenceEvaluator" if implemented else None,
            "reference_implementation": "scenes.workrooms.mechanical_assembly.nist_evaluation.SequenceEvaluator",
            "frozen_path": "evaluator/sequence_evaluator.py", "thresholds": thresholds(task_id),
            "scope": "Native geometry, support and released stability; retries allowed."},
        "source_release": {"path": f"{RELEASE_ROOT}/{task_id}/{release_version}",
                           "manifest_sha256": manifest_sha256} if implemented else None,
        "qualification": {"native_preflight": native_preflight,
            "object_interactions": object_interactions, "manipulation": False,
            "physics_accepted": False, "scene_contact_evidence": deepcopy(list(evidence)),
            "release_status": "prepared_unvalidated" if implemented else "planned",
            "scope": "Deduplicated NIST adaptation. Object feasibility only; hand-object work belongs to data collection."},
        "validation": {**deepcopy(PRIVILEGED_VALIDATION),
            "privileged_entrypoint": "scripts.validation.mechanical_nist_release:run",
            "hand_object_interaction_owner": "data_collection; not a scene release gate",
            "scope": "Native scene/object feasibility only; no model policy or full robot success claim."},
    }
    if not implemented:
        result["cannot_run_reason"] = "Frozen NIST-derived release is not yet selected."
    return result
