from __future__ import annotations

import hashlib
import json
import os

import pytest

from agent.model_client.learning.journal import (
    JournalError, atomic_write_json, canonical_json, read_json,
)
from agent.model_client.learning.skills import SkillLibrary, SkillLibraryError, main
from agent.model_client.outcomes import classify_report


@pytest.fixture
def library(tmp_path):
    return SkillLibrary.create("learner", learning_root=tmp_path, contract={
        "dependencies": {}, "compatibility": {"robot": "test-robot"},
        "rules_sha256": "test-rules", "phase": "practice",
    })


def write(library, path, value):
    return library.write_author(path, value, expected_revision=library.author_revision())


def publish(library, transaction_id="first", **kwargs):
    return library.publish(library.author_revision(), transaction_id=transaction_id,
                           reason="learned from public practice", **kwargs)


def evidence(outcome="success", valid=True):
    return {"source": "bench", "protocol_id": "test-plan-v1", "slot_id": "slot1",
            "attempt_id": "attempt1", "task_outcome": outcome, "valid": valid,
            "execution_status": "completed" if valid else "infra_error", "cost": None}


def select(library, candidate, suffix="1"):
    record = library.record_validation(candidate["event_id"], evidence(), transaction_id="val" + suffix)
    return library.select(candidate["event_id"], validation_event_ids=[record["event_id"]],
        gate={"approved": True, "protocol_id": "test-plan-v1"}, transaction_id="sel" + suffix)


def test_empty_initialization_and_frozen_contract(library):
    assert read_json(library.author / "catalog.json") == {"skills": []}
    assert library.status()["candidate_count"] == 0
    assert library.status()["selected"] is None
    assert library.status()["runtime_isolation_implemented"] is False
    with pytest.raises(FileExistsError):
        SkillLibrary.create("learner", learning_root=library.path.parent, contract={})
    with pytest.raises(SkillLibraryError, match="contract mismatch"):
        SkillLibrary(library.path, expected_contract={})
    atomic_write_json(library.host / "experiment.json", {**library.experiment, "contract": {}})
    with pytest.raises(SkillLibraryError, match="provenance mismatch"):
        SkillLibrary(library.path)


def test_revision_and_phase_write_controls(library):
    old = library.author_revision()
    new = library.write_author("notes/lesson.md", "public observation", expected_revision=old)
    assert new != old
    with pytest.raises(SkillLibraryError, match="stale"):
        library.write_author("notes/lesson.md", "replacement", expected_revision=old)
    for phase in ("validation", "frozen_evaluation"):
        with pytest.raises(SkillLibraryError, match="practice"):
            library.write_author("notes/lesson.md", "test leakage", expected_revision=new, phase=phase)
        with pytest.raises(SkillLibraryError, match="practice"):
            publish(library, phase=phase)
    assert library.delete_author("notes/lesson.md", expected_revision=new) == old
    assert library.journal.read()[-1]["payload"]["operation"] == "delete"


@pytest.mark.parametrize("path", ["../host/secret.json", "/tmp/evil.py", "notes//x.md",
    "notes/../host/x.json", "notes\\x.md", "host/x.json", "skills/x.pickle", "skills/./x.py"])
def test_reject_author_path_escape(library, path):
    with pytest.raises(SkillLibraryError):
        write(library, path, "bad")


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo", "directory_symlink"])
def test_reject_links_and_special_files(library, tmp_path, kind):
    external = tmp_path / "private.txt"
    external.write_text("private")
    target = library.author / "notes" / "leak.txt"
    if kind == "symlink":
        target.symlink_to(external)
    elif kind == "hardlink":
        os.link(external, target)
    elif kind == "fifo":
        os.mkfifo(target)
    else:
        target = library.author / "modules" / "other"
        target.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises((JournalError, OSError)):
        publish(library)
    assert external.read_text() == "private"


def test_publication_snapshots_all_knowledge_and_is_idempotent(library):
    write(library, "notes/lesson.md", "Hypothesis: the public image suggests a corner.")
    revision = library.author_revision()
    first = publish(library)
    bundle_hash = first["payload"]["bundle_hash"]
    manifest = library.verify(bundle_hash)
    assert {row["path"] for row in manifest["files"]} == {"catalog.json", "notes/lesson.md"}
    assert library.publish(revision, transaction_id="first", reason="learned from public practice") == first
    second_origin = publish(library, "same-content", parents=[first["event_id"]])
    assert second_origin["payload"]["bundle_hash"] == bundle_hash
    assert second_origin["event_id"] != first["event_id"]
    write(library, "notes/lesson.md", "Revised hypothesis.")
    second = publish(library, "second", parents=[first["event_id"]])
    assert second["payload"]["bundle_hash"] != bundle_hash
    assert library.read_bundle_file(first["event_id"], "notes/lesson.md").startswith(b"Hypothesis")
    with pytest.raises(SkillLibraryError, match="different request"):
        publish(library, "first")


def test_python_is_parsed_without_execution_and_dependencies_are_closed(library, tmp_path):
    sentinel = tmp_path / "should_not_exist"
    source = f"from pathlib import Path\nPath({str(sentinel)!r}).touch()\ndef run(api, args, state):\n    return state\n"
    write(library, "skills/solution.py", source)
    write(library, "catalog.json", canonical_json({"skills": [{"skill_id": "solution", "module": "skills/solution.py",
        "entrypoint": "run", "description": "agent-written", "tags": [], "input_schema": {}, "output_schema": {}}]}))
    publish(library)
    assert not sentinel.exists()
    write(library, "modules/compute.py", "import unlisted_dependency\n")
    with pytest.raises(SkillLibraryError, match="undeclared"):
        publish(library, "bad-import")
    write(library, "modules/compute.py", "from . import missing\n")
    with pytest.raises(SkillLibraryError, match="missing local"):
        publish(library, "missing-local")
    write(library, "modules/compute.py", "def x(:\n")
    with pytest.raises(SkillLibraryError, match="syntax"):
        publish(library, "syntax")


def test_changed_or_extra_bundle_bytes_are_rejected(library):
    first = publish(library)
    bundle_hash = first["payload"]["bundle_hash"]
    payload = library.bundles / bundle_hash / "payload"
    os.chmod(payload, 0o700)
    (payload / "notes").mkdir()
    (payload / "notes" / "extra.md").write_text("extra knowledge")
    with pytest.raises(SkillLibraryError, match="hash mismatch"):
        library.verify(bundle_hash)


def test_recover_publish_after_prepared_before_rename(library, monkeypatch):
    original = library._complete_prepared
    monkeypatch.setattr(library, "_complete_prepared", lambda event: (_ for _ in ()).throw(RuntimeError("crash")))
    with pytest.raises(RuntimeError):
        publish(library)
    assert library.status()["candidate_count"] == 0
    monkeypatch.setattr(library, "_complete_prepared", original)
    recovered = SkillLibrary(library.path)
    result = recovered.recover()
    assert result["resumed_candidates"] == ["candidate-first"]
    assert recovered.status()["candidate_count"] == 1
    assert recovered.recover()["resumed_candidates"] == []


def test_recover_published_bundle_without_candidate(library, monkeypatch):
    original = library.journal.append
    def crash(kind, *args, **kwargs):
        if kind == "candidate_committed":
            raise RuntimeError("candidate commit lost")
        return original(kind, *args, **kwargs)
    monkeypatch.setattr(library.journal, "append", crash)
    with pytest.raises(RuntimeError):
        publish(library)
    assert len(list(library.bundles.iterdir())) == 1
    recovered = SkillLibrary(library.path)
    assert recovered.recover()["resumed_candidates"] == ["candidate-first"]
    assert publish(recovered)["event_id"] == "candidate-first"


def test_recover_quarantines_unprepared_staging_and_never_adopts_orphan(library, monkeypatch):
    original = library.journal.append
    def crash(kind, *args, **kwargs):
        if kind == "publication_prepared":
            raise RuntimeError("prepared commit lost")
        return original(kind, *args, **kwargs)
    monkeypatch.setattr(library.journal, "append", crash)
    with pytest.raises(RuntimeError):
        publish(library)
    recovered = SkillLibrary(library.path)
    (library.bundles / ("f" * 64)).mkdir()
    result = recovered.recover()
    assert len(result["quarantined_stages"]) == 1
    assert result["orphan_bundles_not_authorized"] == ["f" * 64]
    assert recovered.status()["candidate_count"] == 0
    with pytest.raises(SkillLibraryError, match="provenance"):
        recovered.read_bundle_file("invented", "catalog.json")


def test_infrastructure_evidence_never_becomes_tested_or_selected(library):
    candidate = publish(library)
    before = (library.bundles / candidate["payload"]["bundle_hash"] / "manifest.json").read_bytes()
    record = library.record_validation(candidate["event_id"], evidence("unknown", False), transaction_id="api-down")
    assert not any(event["kind"] == "candidate_tested" for event in library.journal.read())
    with pytest.raises(SkillLibraryError, match="does not qualify"):
        library.select(candidate["event_id"], validation_event_ids=[record["event_id"]],
                       gate={"approved": True, "protocol_id": "test-plan-v1"}, transaction_id="bad-select")
    assert (library.bundles / candidate["payload"]["bundle_hash"] / "manifest.json").read_bytes() == before
    with pytest.raises(SkillLibraryError, match="unknown"):
        library.record_validation(candidate["event_id"], evidence("unknown", True), transaction_id="bad-evidence")


def continuous_evidence(height=0.3, **report_changes):
    report = {
        "termination": "server_finished", "result_source": "server_result",
        "result": {"outcome": "sim_timeout", "physics_valid": True, "scoring": {
            "schema": "terminal_task_scoring.v1", "mode": "continuous", "objective": "maximize",
            "score": height, "score_unit": "m", "score_valid": True,
            "metrics": {"best_height_m": height}, "metric_units": {"best_height_m": "m"}}},
        **report_changes,
    }
    result = classify_report(report)
    return {**evidence("unknown", result["result_valid_for_scoring"]), **result,
            "execution_status": result["run_status"]}


@pytest.mark.parametrize("height", [0.0, 0.025, 0.3])
def test_continuous_score_is_valid_candidate_evidence_without_binary_success(library, height):
    candidate = publish(library)
    measured = continuous_evidence(height)
    record = library.record_validation(candidate["event_id"], measured, transaction_id="height")
    assert record["payload"]["evidence"] == measured
    assert measured["task_outcome"] == "unknown" and measured["task_success"] is None
    assert measured["task_score"] == height
    assert any(event["kind"] == "candidate_tested" for event in library.journal.read())
    # The host's declared gate decides selection, including whether a zero score
    # meets its protocol. The library must not invent a success threshold.
    selected = library.select(candidate["event_id"], validation_event_ids=[record["event_id"]],
        gate={"approved": True, "protocol_id": "test-plan-v1"}, transaction_id="choose")
    assert selected["candidate_event_id"] == candidate["event_id"]
    reopened = SkillLibrary(library.path, expected_contract=library.contract)
    assert reopened.selected() == selected
    assert read_json(library.host / "evidence" / (record["event_id"] + ".json")) == record


def test_continuous_score_survives_normal_model_output_budget(library):
    candidate = publish(library)
    measured = continuous_evidence(
        termination="client_output_budget", error_type="ModelGatewayError",
        run_error={"error_category": "output_budget_exhausted"})
    assert measured["execution_status"] == "agent_budget" and measured["valid"]
    library.record_validation(candidate["event_id"], measured, transaction_id="budget")
    assert any(event["kind"] == "candidate_tested" for event in library.journal.read())


def test_missing_continuous_score_is_retained_only_as_invalid_audit_evidence(library):
    candidate = publish(library)
    measured = continuous_evidence(result={"outcome": "sim_timeout", "physics_valid": True, "scoring": {}})
    assert measured["task_score"] is None and measured["task_score_unit"] is None
    assert not measured["valid"]
    record = library.record_validation(candidate["event_id"], measured, transaction_id="missing")
    assert record["payload"]["evidence"] == measured
    assert not any(event["kind"] == "candidate_tested" for event in library.journal.read())


@pytest.mark.parametrize("changes", [
    {"error_type": "ModelGatewayError", "run_error": {"error_category": "transport_timeout"}},
    {"action_recovery_pending": True},
    {"finalization_error": {"error_type": "BenchHTTPError", "error_category": "transport_error"}},
])
def test_continuous_audit_score_from_unresolved_execution_cannot_select(library, changes):
    candidate = publish(library)
    measured = continuous_evidence(**changes)
    record = library.record_validation(candidate["event_id"], measured, transaction_id="interrupted")
    assert record["payload"]["evidence"]["task_score"] == 0.3
    assert not measured["valid"] and not measured["task_score_valid"]
    assert not any(event["kind"] == "candidate_tested" for event in library.journal.read())
    with pytest.raises(SkillLibraryError, match="does not qualify"):
        library.select(candidate["event_id"], validation_event_ids=[record["event_id"]],
            gate={"approved": True, "protocol_id": "test-plan-v1"}, transaction_id="choose")
    with pytest.raises(SkillLibraryError, match="cannot qualify"):
        library.record_validation(candidate["event_id"], {**measured, "valid": True},
                                  transaction_id="false-validity")


@pytest.mark.parametrize("changes", [
    {"task_score": None}, {"task_score": float("nan")}, {"task_score": "0.3"},
    {"task_score": True}, {"task_score_unit": None}, {"task_score_unit": ""},
    {"task_success": True}, {"task_outcome": "success"}, {"task_score_valid": 1},
    {"result_valid_for_scoring": False}, {"execution_status": "infra_error"},
])
def test_reject_malformed_or_binary_continuous_validation(library, changes):
    candidate = publish(library)
    with pytest.raises(SkillLibraryError, match="continuous"):
        library.record_validation(candidate["event_id"], {**continuous_evidence(), **changes},
                                  transaction_id="bad-score")
    assert not any(event["kind"] in ("validation_recorded", "candidate_tested")
                   for event in library.journal.read())


def test_continuous_mode_alone_does_not_make_unknown_evidence_valid(library):
    candidate = publish(library)
    with pytest.raises(SkillLibraryError, match="continuous"):
        library.record_validation(candidate["event_id"],
            {**evidence("unknown", True), "task_scoring_mode": "continuous"},
            transaction_id="missing-score")


def test_selected_pointer_rebuilt_from_chain_and_invalid_cache_rejected(library):
    first = publish(library)
    old = select(library, first)
    (library.host / "selected.json").unlink()
    assert library.selected() == old
    write(library, "notes/revision.md", "new public feedback")
    second = publish(library, "second", parents=[first["event_id"]])
    new = select(library, second, "2")
    atomic_write_json(library.host / "selected.json", old)
    assert library.selected() == new
    atomic_write_json(library.host / "selected.json", {**new, "bundle_hash": "f" * 64})
    with pytest.raises(SkillLibraryError, match="provenance"):
        library.selected()


def test_identical_content_does_not_inherit_candidate_qualification(library):
    first = publish(library)
    validation = library.record_validation(first["event_id"], evidence(), transaction_id="v1")
    second = publish(library, "second")
    assert first["payload"]["bundle_hash"] == second["payload"]["bundle_hash"]
    with pytest.raises(SkillLibraryError, match="does not qualify"):
        library.select(second["event_id"], validation_event_ids=[validation["event_id"]],
                       gate={"approved": True, "protocol_id": "test-plan-v1"}, transaction_id="s1")


def test_explicit_import_preserves_history_but_not_validation_rights(library, tmp_path):
    write(library, "notes/public.md", "public practice observation")
    candidate = publish(library)
    select(library, candidate)
    target = SkillLibrary.create("target", learning_root=tmp_path, contract={"dependencies": {}})
    provenance = {"scenes": ["practice-scene"], "phase": "practice", "cost": None, "licenses": ["agent-generated"]}
    imported = target.import_candidate(library, candidate["event_id"], transaction_id="import1", provenance=provenance)
    assert imported["payload"]["bundle_hash"] == candidate["payload"]["bundle_hash"]
    assert target.read_author("catalog.json") == canonical_json({"skills": []})
    assert target.status()["selected"] is None
    origin_hash = imported["payload"]["import"]["origin_sha256"]
    origin = read_json(target.host / "origins" / (origin_hash + ".json"))
    assert hashlib.sha256(canonical_json(origin)).hexdigest() == origin_hash
    assert candidate in origin["journal"]
    assert any(event["kind"] == "validation_recorded" for event in origin["journal"])
    assert target.import_candidate(library, candidate["event_id"], transaction_id="import1", provenance=provenance) == imported
    with pytest.raises(SkillLibraryError, match="evaluation"):
        target.import_candidate(library, candidate["event_id"], transaction_id="import2",
                                provenance={**provenance, "phase": "frozen_evaluation"})


def test_cli_roundtrip(tmp_path, capsys):
    contract = tmp_path / "contract.json"
    atomic_write_json(contract, {"dependencies": {}})
    assert main(["init", "cli", "--learning-root", str(tmp_path), "--contract", str(contract)]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["candidate_count"] == 0
    assert main(["publish", str(tmp_path / "cli"), "--revision", status["author_revision"],
                 "--transaction-id", "cli1", "--reason", "empty baseline"]) == 0
    candidate = json.loads(capsys.readouterr().out)
    assert main(["verify", str(tmp_path / "cli"), candidate["payload"]["bundle_hash"]]) == 0
    assert json.loads(capsys.readouterr().out)["catalog"] == {"skills": []}


def test_old_selection_retry_returns_original_receipt_without_switching_version(library):
    first = publish(library)
    original = select(library, first, "1")
    write(library, "notes/updated.md", "new hypothesis")
    second = publish(library, "second")
    current = select(library, second, "2")
    retry = library.select(first["event_id"], validation_event_ids=["validation-val1"],
        gate={"approved": True, "protocol_id": "test-plan-v1"}, transaction_id="sel1")
    assert retry == original
    assert library.selected() == current


def test_recovery_seals_directory_after_rename_crash(library, monkeypatch):
    import agent.model_client.learning.skills as module
    original = module.rename_noreplace
    def rename_then_crash(source, destination):
        original(source, destination)
        if destination.parent == library.bundles:
            raise RuntimeError("crash immediately after publication rename")
    monkeypatch.setattr(module, "rename_noreplace", rename_then_crash)
    with pytest.raises(RuntimeError):
        publish(library)
    bundle = next(library.bundles.iterdir())
    assert bundle.stat().st_mode & 0o200
    reopened = SkillLibrary(library.path)
    assert reopened.recover()["resumed_candidates"] == ["candidate-first"]
    assert not bundle.stat().st_mode & 0o222


def test_transitive_import_includes_full_ancestry_and_known_evidence(library, tmp_path):
    first = publish(library)
    first_validation = library.record_validation(first["event_id"], {**evidence(), "cost": {"tokens": 12}},
                                                 transaction_id="first-v")
    provenance = {"scenes": ["practice"], "phase": "practice", "cost": None, "licenses": []}
    second = SkillLibrary.create("second", learning_root=tmp_path, contract={})
    middle = second.import_candidate(library, first["event_id"], transaction_id="from-first", provenance=provenance)
    third = SkillLibrary.create("third", learning_root=tmp_path, contract={})
    imported = third.import_candidate(second, middle["event_id"], transaction_id="from-second", provenance=provenance)
    origin_hash = imported["payload"]["import"]["origin_sha256"]
    origin = read_json(third.host / "origins" / (origin_hash + ".json"))
    ancestor = origin["ancestor_origins"][middle["payload"]["import"]["origin_sha256"]]
    assert first_validation in ancestor["journal"]
    assert ancestor["journal"][first_validation["sequence"] - 1]["payload"]["evidence"]["cost"]["tokens"] == 12
    # New source history must not change an already committed import's receipt.
    second.record_validation(middle["event_id"], evidence(), transaction_id="later")
    assert third.import_candidate(second, middle["event_id"], transaction_id="from-second", provenance=provenance) == imported
    # Neither a missing origin nor changed ancestor history may be silently trusted.
    origin["ancestor_origins"].clear()
    atomic_write_json(third.host / "origins" / (origin_hash + ".json"), origin)
    with pytest.raises(SkillLibraryError, match="origin hash"):
        third.read_bundle_file(imported["event_id"], "catalog.json")


def test_descendant_candidate_requires_preserved_parent_origins(library, tmp_path):
    source_candidate = publish(library)
    target = SkillLibrary.create("descendant", learning_root=tmp_path, contract={})
    imported = target.import_candidate(library, source_candidate["event_id"], transaction_id="import1",
        provenance={"scenes": [], "phase": "practice", "cost": None, "licenses": []})
    descendant = publish(target, "descendant", parents=[imported["event_id"]])
    origin_hash = imported["payload"]["import"]["origin_sha256"]
    (target.host / "origins" / (origin_hash + ".json")).unlink()
    with pytest.raises(FileNotFoundError):
        target.read_bundle_file(descendant["event_id"], "catalog.json")
