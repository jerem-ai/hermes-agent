"""Read-only status distinguishes settings, usage receipts and observed outcomes."""
import argparse
import json
import sqlite3

from hermes_cli import learning_status


def _config():
    return {"model": {"provider": "custom", "default": "leyline/tars", "base_url": "http://127.0.0.1:8790/v1"},
            "memory": {"memory_enabled": True, "user_profile_enabled": True, "nudge_interval": 10},
            "skills": {"creation_nudge_interval": 15}, "auxiliary": {"background_review": {"enabled": True}}}


def test_enabled_and_usage_are_not_completed_reviews(tmp_path):
    db = tmp_path / "state.db"
    with sqlite3.connect(db) as conn:
        conn.execute("create table session_model_usage(session_id text, task text, api_call_count integer)")
        conn.execute("insert into session_model_usage values('s', 'background_review', 3)")
    before = db.read_bytes()
    result = learning_status.collect_status(tmp_path, _config(), session_id="s")
    assert result["configuration"]["background_review_enabled"] is True
    assert result["observed"]["outcomes"] == {}
    assert result["observed"]["usage_receipts"] == 1
    assert result["observed"]["outcome_evidence"] == "not_recorded"
    assert db.read_bytes() == before
    assert not (tmp_path / "logs").exists()


def test_nonmanaged_leyline_auto_is_not_idle_queued(tmp_path, monkeypatch):
    monkeypatch.setattr(learning_status, "_managed_netloc", lambda: "localhost:9999")
    result = learning_status.collect_status(tmp_path, _config())
    assert result["configuration"]["defer_mode"] == "auto"
    assert result["configuration"]["configured_dispatch"] == "after_turn"
    assert result["observed"]["outcome_evidence"] == "not_recorded"


def test_status_closes_its_read_only_database(tmp_path, monkeypatch):
    path = tmp_path / "state.db"
    conn = sqlite3.connect(path)
    conn.execute("create table session_model_usage(session_id text, task text)")
    conn.close()
    closed = []
    class TrackedConnection(sqlite3.Connection):
        def close(self):
            closed.append(True)
            super().close()
    connect = sqlite3.connect
    monkeypatch.setattr(learning_status.sqlite3, "connect",
                        lambda *a, **kw: connect(*a, **kw, factory=TrackedConnection))
    assert learning_status.collect_status(tmp_path, _config())["observed"]["usage_receipts"] == 0
    assert closed == [True]


def test_managed_endpoint_and_unicode_profile_are_valid(tmp_path, monkeypatch):
    home = tmp_path / "profile 雪"
    home.mkdir()
    cfg = _config()
    cfg["auxiliary"]["background_review"]["base_url"] = "http://LOCALHOST:9999/v1"
    cfg["auxiliary"]["background_review"]["provider"] = "custom"
    monkeypatch.setattr(learning_status, "_managed_netloc", lambda: "localhost:9999")
    result = learning_status.collect_status(home, cfg)
    assert result["profile_home"] == str(home)
    assert result["configuration"]["configured_dispatch"] == "idle_queue"
    assert result["observed"]["outcomes"] == {}
    assert sorted(home.iterdir()) == []


def test_failed_cancelled_and_staged_are_visible_without_content(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    rows = [
        {"version": 1, "time": "2026-10-20T00:00:00Z", "session_id": "s", "review_id": "one", "event": "failed", "error_type": "RuntimeError"},
        {"version": 1, "time": "2026-10-20T00:01:00Z", "session_id": "s", "review_id": "two", "event": "cancelled", "writes": {"memory": 1, "skill": 0, "staged": 1}},
        {"version": 1, "time": "2026-10-20T00:02:00Z", "session_id": "other", "review_id": "three", "event": "completed", "content": "PRIVATE"},
    ]
    (logs / "learning-reviews.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n{partial")
    result = learning_status.collect_status(tmp_path, _config(), session_id="s")
    assert result["observed"]["outcomes"] == {"failed": 1, "cancelled": 1}
    assert result["observed"]["successful_write_results"] == {"memory": 1, "skill": 0, "staged": 1}
    assert result["observed"]["unreadable_records"] == 1
    assert "PRIVATE" not in json.dumps(result)


def test_started_event_is_not_an_observed_outcome(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "learning-reviews.jsonl").write_text(json.dumps({"event": "started", "review_id": "r"}) + "\n")
    observed = learning_status.collect_status(tmp_path, _config())["observed"]
    assert observed["outcome_evidence"] == "not_recorded"
    assert observed["recent"][0]["event"] == "started"


def test_cli_parser_and_json_use_active_profile(tmp_path, monkeypatch, capsys):
    from hermes_cli.commands import resolve_command
    from hermes_cli.subcommands.memory import build_memory_parser
    from hermes_cli import memory_setup
    parser = argparse.ArgumentParser()
    build_memory_parser(parser.add_subparsers(), cmd_memory=memory_setup.memory_command)
    args = parser.parse_args(["memory", "status", "--learning", "--json", "--session", "s"])
    monkeypatch.setattr(learning_status, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", _config)
    args.func(args)
    result = json.loads(capsys.readouterr().out)
    assert result["profile_home"] == str(tmp_path)
    assert result["session_id"] == "s"
    assert "status" in resolve_command("memory").subcommands


def test_native_memory_status_does_not_require_or_write_a_memory_store(tmp_path, monkeypatch):
    from tools import memory_tool
    monkeypatch.setattr(memory_tool, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", _config)
    before = set(tmp_path.rglob("*"))
    result = json.loads(memory_tool.memory_tool(action="status", target="user"))
    assert result["success"]
    assert result["profile_home"] == str(tmp_path)
    assert set(tmp_path.rglob("*")) == before
    mixed = json.loads(memory_tool.memory_tool(action="status", content="do not silently discard this edit"))
    assert mixed["success"] is False
    assert set(tmp_path.rglob("*")) == before


def test_shared_slash_status_is_read_only_and_accepts_session_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(learning_status, "get_hermes_home", lambda: tmp_path)
    monkeypatch.setattr("hermes_cli.config.load_config_readonly", _config)
    before = set(tmp_path.rglob("*"))
    result = json.loads(learning_status.handle_slash(["status", "--json", "--session", "s 雪"]))
    assert result["session_id"] == "s 雪"
    assert set(tmp_path.rglob("*")) == before
    assert learning_status.handle_slash(["approve", "all"]) is None
    assert "Usage:" in learning_status.handle_slash(["status", "--limit", "nonsense"])
