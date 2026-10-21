"""The roadmap incident: passing old tests and editing docs cannot complete implementation."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from agent.acceptance_contract import (
    canonical, context, digest, directory, enforce_final, fold_snapshots, guarded,
    load, register, report,
)
from agent.acceptance_verification import check_receipt, verify


def make_spec(tmp_path, *, root_type="implementation", behavior=False):
    subject = tmp_path / "Crème 雪.txt"
    subject.write_text("correct unusual documentation")
    script = tmp_path / "verify.py"
    script.write_text(
        "import json,pathlib,sys\n"
        "print(json.dumps({'passed': pathlib.Path(sys.argv[1]).read_text() == "
        "'correct unusual documentation'}))\n"
    )
    verifier = {
        "argv": [sys.executable, str(script), str(subject)],
        "files": {str(p): digest(p.read_bytes()) for p in (script, subject)},
        "maximum_age_seconds": 60,
    }
    checks = [{"id": "saved-behavior", "description": "Actual caller behavior accepted",
               "kind": "behavior", "verifier": verifier if behavior else None}]
    if root_type == "documentation":
        checks = [{"id": "doc", "description": "Documentation checked",
                   "kind": "documentation", "verifier": verifier}]
    return {"schema": "hermes.acceptance-contract.v1", "root": "implementation",
            "tasks": [
                {"id": "implementation", "objective": "Deliver working caller acceptance",
                 "type": root_type, "checks": checks},
                {"id": "docs", "parent": "implementation", "objective": "Correct the roadmap",
                 "type": "documentation", "checks": [
                     {"id": "doc", "kind": "documentation", "description": "Roadmap facts verified",
                      "verifier": verifier}]}]}


@pytest.fixture
def agent(tmp_path):
    return SimpleNamespace(session_id="session-a",
                           _session_db=SimpleNamespace(db_path=tmp_path / "state.db",
                                                       get_session=lambda sid: None))


def attach(agent, spec):
    sha = register(agent.session_id, spec, root_dir=directory(agent))
    return sha


def test_doc_and_existing_tests_do_not_finish_parent(agent, tmp_path):
    spec = make_spec(tmp_path)
    sha = attach(agent, spec)
    verify(directory(agent), spec, sha, "docs", "doc")
    assert report(agent)["tasks"]["docs"]["complete"]
    assert not report(agent)["tasks"]["implementation"]["complete"]
    row = {}
    text, blocked = enforce_final(agent, "Yes, the original task is complete. 687 tests pass.", row)
    assert blocked and "remains open" in text
    assert row["acceptance_contract"]["complete"] is False
    # A precisely scoped child completion preserves the standing objective.
    text = "docs is complete. The parent objective remains open."
    assert enforce_final(agent, text, {}) == (text, False)


def test_legitimate_documentation_assignment_can_complete(agent, tmp_path):
    spec = make_spec(tmp_path, root_type="documentation")
    sha = attach(agent, spec)
    for task, check in [("implementation", "doc"), ("docs", "doc")]:
        assert verify(directory(agent), spec, sha, task, check)["passed"]
    assert report(agent)["tasks"]["implementation"]["complete"]
    assert enforce_final(agent, "Everything is done.", {}) == ("Everything is done.", False)


def test_implementation_requires_behavior_not_just_tests(agent, tmp_path):
    spec = make_spec(tmp_path)
    spec["tasks"][0]["checks"][0]["kind"] = "tests"
    with pytest.raises(ValueError, match="behavior acceptance"):
        attach(agent, spec)


def test_contract_parent_cannot_be_replaced_by_doc_subtask(agent, tmp_path):
    spec = make_spec(tmp_path)
    attach(agent, spec)
    different = copy.deepcopy(spec)
    different["tasks"][0]["objective"] = "Only update the roadmap"
    with pytest.raises(ValueError, match="parent contract"):
        attach(agent, different)


def test_fresh_subject_bound_behavior_pass_and_changed_subject_refusal(agent, tmp_path):
    spec = make_spec(tmp_path, behavior=True)
    sha = attach(agent, spec)
    for task, check in [("implementation", "saved-behavior"), ("docs", "doc")]:
        verify(directory(agent), spec, sha, task, check)
    assert report(agent)["tasks"]["implementation"]["complete"]
    Path(spec["tasks"][0]["checks"][0]["verifier"]["argv"][-1]).write_text("changed after verification")
    assert not report(agent)["tasks"]["implementation"]["complete"]


def test_nonzero_exit_cannot_be_hidden_by_success_output(agent, tmp_path):
    spec = make_spec(tmp_path)
    check = spec["tasks"][1]["checks"][0]
    program = Path(check["verifier"]["argv"][1])
    program.write_text("import json,sys\nprint(json.dumps({'passed':True}))\nsys.exit(9)\n")
    check["verifier"]["files"][str(program)] = digest(program.read_bytes())
    sha = attach(agent, spec)
    receipt = verify(directory(agent), spec, sha, "docs", "doc")
    assert receipt["exit_code"] == 9 and receipt["passed"] is False
    assert not check_receipt(directory(agent), sha, "docs", check)


def test_restart_and_compaction_keep_root_at_production_scale(agent, tmp_path):
    attach(agent, make_spec(tmp_path))
    history = [{"role": "user", "content": "Original assignment " + "雪" * 100_000},
               {"role": "assistant", "content": "Reading the current source."}]
    # Compression produces a new candidate history; original cached history stays intact.
    compressed = [{"role": "user", "content": "Only the latest documentation subtask survived."}]
    from tools.todo_tool import TodoStore
    from agent.conversation_compression import _fold_todo_snapshot
    agent._todo_store = TodoStore()
    fold_snapshots(agent, compressed, _fold_todo_snapshot)
    restored = SimpleNamespace(session_id=agent.session_id, _session_db=agent._session_db)
    assert "Actual caller behavior accepted" in context(restored)
    assert "Deliver working caller acceptance" in compressed[-1]["content"]
    assert len(history[0]["content"]) > 100_000
    assert not report(restored)["tasks"]["implementation"]["complete"]


def test_descendant_session_and_other_profile_isolation(agent, tmp_path):
    attach(agent, make_spec(tmp_path))
    child = SimpleNamespace(session_id="child", _session_db=SimpleNamespace(
        db_path=agent._session_db.db_path,
        get_session=lambda sid: {"parent_session_id": "session-a"} if sid == "child" else None))
    assert load(child) == load(agent)
    other = SimpleNamespace(session_id=agent.session_id, _session_db=SimpleNamespace(
        db_path=tmp_path / "other-profile/state.db", get_session=lambda sid: None))
    assert not guarded(other) and report(other) is None


def test_unreadable_contract_refuses_completion(agent, tmp_path):
    sha = attach(agent, make_spec(tmp_path))
    (directory(agent) / (sha + ".json")).write_text("{}")
    assert guarded(agent)
    assert enforce_final(agent, "Everything is done.", {})[1]


def test_session_db_error_during_lineage_walk_reads_as_unreadable(agent, tmp_path):
    import sqlite3
    from agent.acceptance_contract import HEADER

    attach(agent, make_spec(tmp_path))

    def locked(_sid):
        raise sqlite3.OperationalError("database is locked")

    child = SimpleNamespace(session_id="child", _session_db=SimpleNamespace(
        db_path=agent._session_db.db_path, get_session=locked))
    text = context(child)
    assert text.startswith(HEADER) and "unreadable" in text


def test_profile_without_a_contract_store_never_reads_the_session_db(tmp_path):
    def must_not_read(_sid):
        raise AssertionError("no contract store, so no lineage walk")

    bare = SimpleNamespace(session_id="s", _session_db=SimpleNamespace(
        db_path=tmp_path / "state.db", get_session=must_not_read))
    assert load(bare) is None
    assert report(bare) is None


def test_expired_receipt_does_not_complete_task(agent, tmp_path, monkeypatch):
    spec = make_spec(tmp_path)
    sha = attach(agent, spec)
    receipt = verify(directory(agent), spec, sha, "docs", "doc")
    monkeypatch.setattr("agent.acceptance_verification.time.time",
                        lambda: receipt["observed_at"] + 61)
    assert not report(agent)["tasks"]["docs"]["complete"]


def test_literal_completion_quote_is_a_valid_answer(agent, tmp_path):
    attach(agent, make_spec(tmp_path))
    text = 'The message says "Everything is done." I still need to verify it.'
    assert enforce_final(agent, text, {}) == (text, False)
    text = "The original task is not complete."
    assert enforce_final(agent, text, {}) == (text, False)


def test_inline_code_cannot_substitute_for_pinned_verifier(agent, tmp_path):
    spec = make_spec(tmp_path)
    spec["tasks"][1]["checks"][0]["verifier"]["argv"] = [
        sys.executable, "-c", "print(123)", str(tmp_path / "Crème 雪.txt")]
    with pytest.raises(ValueError, match="source-pinned"):
        attach(agent, spec)


def test_contract_compression_does_not_erase_a_literal_user_header(agent, tmp_path):
    attach(agent, make_spec(tmp_path))
    from agent.acceptance_contract import HEADER
    original = "Explain this literal: " + HEADER + " and preserve this instruction."
    compressed = [{"role": "user", "content": original}]
    fold_snapshots(agent, compressed, lambda *args: None)
    assert compressed[0]["content"].startswith(original)


def test_reviewed_verifier_binding_preserves_parent_and_invalidates_old_receipts(agent, tmp_path):
    from agent.acceptance_contract import bind_verifier
    spec = make_spec(tmp_path)
    sha = attach(agent, spec)
    verify(directory(agent), spec, sha, "docs", "doc")
    definition = spec["tasks"][1]["checks"][0]["verifier"]
    new_sha = bind_verifier(agent, sha, "implementation", "saved-behavior", definition)
    rebound, actual = load(agent)
    assert actual == new_sha and actual != sha
    assert rebound["tasks"][0]["objective"] == spec["tasks"][0]["objective"]
    assert not report(agent)["tasks"]["implementation"]["complete"]
    assert not report(agent)["tasks"]["docs"]["complete"]
    with pytest.raises(ValueError, match="parent contract changed"):
        bind_verifier(agent, sha, "implementation", "saved-behavior", definition)


def test_broad_completion_is_not_excused_by_an_open_parent_footer(agent, tmp_path):
    spec = make_spec(tmp_path)
    sha = attach(agent, spec)
    verify(directory(agent), spec, sha, "docs", "doc")
    text = "Everything is done. docs is complete. The parent objective remains open."
    assert enforce_final(agent, text, {})[1]


def test_forged_cached_pass_is_rechecked_by_harness(agent, tmp_path):
    from agent.acceptance_contract import atomic_json
    from agent.acceptance_verification import receipt_path
    import time
    spec = make_spec(tmp_path, behavior=True)
    spec["tasks"] = spec["tasks"][:1]
    check = spec["tasks"][0]["checks"][0]
    program = Path(check["verifier"]["argv"][1])
    program.write_text("import json\nprint(json.dumps({'passed':False}))\n")
    check["verifier"]["files"][str(program)] = digest(program.read_bytes())
    sha = attach(agent, spec)
    atomic_json(receipt_path(directory(agent), sha, "implementation", check["id"]), {
        "schema": "hermes.acceptance-receipt.v1", "contract": sha,
        "task": "implementation", "check": check["id"], "kind": "behavior",
        "passed": True, "exit_code": 0, "observed_at": time.time(),
        "verifier_sha256": digest(canonical(check["verifier"])),
    })
    assert report(agent)["tasks"]["implementation"]["complete"]
    assert enforce_final(agent, "Everything is done.", {})[1]
    assert not report(agent)["tasks"]["implementation"]["complete"]
