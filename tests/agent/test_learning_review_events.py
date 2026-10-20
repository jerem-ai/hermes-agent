"""Learning outcomes must describe execution and applied writes, never model claims."""
import json
import threading
from types import SimpleNamespace

from agent import learning_review_events as events


def _call(name, args, result, call_id="new"):
    return [
        {"role": "assistant", "tool_calls": [{"id": call_id, "function": {
            "name": name, "arguments": json.dumps(args)}}]},
        {"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)},
    ]


def test_failed_review_cannot_be_recorded_as_completed():
    assert events.review_outcome({"completed": False, "partial": True}, False) == "incomplete"
    assert events.review_outcome({"completed": True, "error": "failed"}, False) == "failed"
    assert events.review_outcome(None, False) == "unknown"
    assert events.review_outcome({"completed": True}, True) == "cancelled"


def test_applied_unicode_batch_survives_long_history_and_incomplete_review():
    old = _call("memory", {"action": "add"}, {"success": True}, "old")
    history = [{"role": "user", "content": "quoted 雪 https://example.test/a?b=c\n" * 9000}] + old
    new = _call("skill_manage", {"action": "batch"}, {
        "success": True, "operations_applied": True, "results": [
            {"success": True, "action": "create", "name": "unicode-fixture"},
            {"success": False, "action": "patch", "name": "not-applied"},
        ]})
    messages = history + new + [{"role": "assistant", "content": "I saved everything."}]
    assert events.write_counts(messages, history) == {"memory": 0, "skill": 1, "staged": 0}
    assert events.review_outcome({"completed": False, "partial": True}, False) == "incomplete"


def test_staged_and_rolled_back_writes_are_not_applied():
    staged = _call("memory", {"action": "add"}, {"success": True, "staged": True})
    rollback = _call("skill_manage", {"action": "batch"}, {
        "success": True, "operations_applied": False,
        "results": [{"success": True, "action": "create"}],
    }, "rolled-back")
    assert events.write_counts(staged + rollback, []) == {"memory": 0, "skill": 0, "staged": 1}
    assert events.write_counts([{"role": "assistant", "content": "Memory updated."}], []) == {
        "memory": 0, "skill": 0, "staged": 0}
    duplicate = _call("memory", {"action": "add"}, {
        "success": True, "message": "Entry already exists (no duplicate added)."})
    assert events.write_counts(duplicate, [])["memory"] == 0


def test_event_is_profile_scoped_and_excludes_private_text(tmp_path, monkeypatch):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    default = tmp_path / "default"
    profile = tmp_path / "texting"
    parent = SimpleNamespace(session_id="session-snow", _session_db=SimpleNamespace(db_path=default / "state.db"))
    token = set_hermes_home_override(str(profile))
    try:
        events.record_event(parent, "review-1", "failed", error_type="RuntimeError")
    finally:
        reset_hermes_home_override(token)
    rows = [json.loads(x) for x in (profile / "logs/learning-reviews.jsonl").read_text().splitlines()]
    assert rows[0]["event"] == "failed"
    assert rows[0]["session_id"] == "session-snow"
    assert rows[0]["error_type"] == "RuntimeError"
    assert not (default / "logs").exists()
    assert set(rows[0]) <= {"version", "time", "session_id", "review_id", "event", "error_type", "reason", "writes"}


def test_event_failure_does_not_abort_foreground(tmp_path, monkeypatch, caplog):
    parent = SimpleNamespace(session_id="s", _session_db=SimpleNamespace(db_path=tmp_path / "state.db"))
    monkeypatch.setattr(events, "_append", lambda *_: (_ for _ in ()).throw(OSError("private diagnostic")))
    events.record_event(parent, "r", "started")
    assert "learning review event" in caplog.text.lower()
    assert "private diagnostic" not in caplog.text


def _run_scripted_review(tmp_path, monkeypatch, *, outcome=None, raises=False, cancel=False, save=True):
    from agent import background_review as br
    from tools.memory_tool import MemoryStore
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    parent = SimpleNamespace(session_id="parent", provider="custom", _active_children=[],
                             _active_children_lock=threading.Lock(), memory_notifications="off",
                             background_review_callback=None, _safe_print=lambda *_: None,
                             _emit_auxiliary_failure=lambda *_: None,
                             _session_db=SimpleNamespace(db_path=tmp_path / "state.db"))
    run = br.prepare_background_review_run(parent)
    history = [{"role": "user", "content": "quoted 雪 correction\n" * 14000}]
    class Child:
        model = "mocked-model"
        provider = "custom"
        base_url = "http://127.0.0.1:1/v1"
        _session_messages = []
        def run_conversation(self, **kwargs):
            store = MemoryStore()
            store.load_from_disk()
            frozen = store.format_for_system_prompt("memory")
            self._session_messages = list(history)
            if save:
                result = store.add("memory", "Use the active profile: quoted 雪.")
                self._session_messages += _call("memory", {"action": "add"}, result)
                assert store.format_for_system_prompt("memory") == frozen
            if cancel:
                run.cancel_requested.set()
            if raises:
                raise RuntimeError("PRIVATE provider payload")
            return outcome
        def release_clients(self):
            pass
    monkeypatch.setattr(br, "build_cache_parity_fork", lambda *_args, **_kwargs: (Child(), {}, False))
    monkeypatch.setattr(br, "_review_tool_whitelist", lambda *_: (set(), set()))
    monkeypatch.setattr(br, "_parent_can_emit_tool_calls", lambda *_: True)
    token = set_hermes_home_override(str(tmp_path))
    try:
        br._run_review_in_thread(parent, history, "fixture", review_run=run, review_memory=True)
        fresh = MemoryStore()
        fresh.load_from_disk()
        if save:
            assert "quoted 雪" in fresh.format_for_system_prompt("memory")
    finally:
        reset_hermes_home_override(token)
    path = events.event_path(tmp_path)
    assert path.exists(), "A review returned, but no execution outcome was recorded"
    return [json.loads(x) for x in path.read_text().splitlines()]


def test_real_worker_records_incomplete_after_a_persisted_correction(tmp_path, monkeypatch):
    rows = _run_scripted_review(tmp_path, monkeypatch, outcome={"completed": False, "partial": True})
    assert rows[-1]["event"] == "incomplete"
    assert rows[-1]["writes"] == {"memory": 1, "skill": 0, "staged": 0}
    assert "completed" not in [r["event"] for r in rows]


def test_real_worker_keeps_partial_write_on_exception_without_private_text(tmp_path, monkeypatch):
    rows = _run_scripted_review(tmp_path, monkeypatch, raises=True)
    assert rows[-1]["event"] == "failed"
    assert rows[-1]["writes"]["memory"] == 1
    assert rows[-1]["error_type"] == "RuntimeError"
    assert "PRIVATE" not in json.dumps(rows)


def test_real_worker_accepts_cancelled_review_that_already_saved_a_correction(tmp_path, monkeypatch):
    rows = _run_scripted_review(tmp_path, monkeypatch, outcome={"completed": True}, cancel=True)
    assert rows[-1]["event"] == "cancelled"
    assert rows[-1]["writes"]["memory"] == 1


def test_completed_review_can_correctly_save_nothing(tmp_path, monkeypatch):
    rows = _run_scripted_review(tmp_path, monkeypatch, outcome={"completed": True}, save=False)
    assert rows[-1]["event"] == "completed"
    assert rows[-1]["writes"] == {"memory": 0, "skill": 0, "staged": 0}


def test_tool_incapable_runtime_releases_review_fence(tmp_path, monkeypatch):
    from agent import background_review as br
    parent = SimpleNamespace(session_id="blocked-runtime", provider="test", _active_children=[],
                             _active_children_lock=threading.Lock(),
                             _session_db=SimpleNamespace(db_path=tmp_path / "state.db"))
    run = br.prepare_background_review_run(parent)
    monkeypatch.setattr(br, "_parent_can_emit_tool_calls", lambda *_: False)
    monkeypatch.setattr(br, "_resolve_review_runtime", lambda *_: {"routed": False})
    br._run_review_in_thread(parent, [], "fixture", review_run=run)
    assert run.request_done.is_set(), "A skipped review must not leave the next turn waiting on its fence"
    assert parent._background_review_run is None
