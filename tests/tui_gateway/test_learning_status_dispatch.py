"""Desktop fallback dispatch must inspect its own profile without loading a store."""
import json

from tui_gateway import server


def test_desktop_status_dispatch_uses_routed_profile(tmp_path, monkeypatch):
    root, profile = tmp_path / "launch", tmp_path / "launch/profiles/texting"
    root.mkdir()
    profile.mkdir(parents=True)
    (root / "config.yaml").write_text("auxiliary:\n  background_review:\n    enabled: false\n")
    (profile / "config.yaml").write_text("auxiliary:\n  background_review:\n    enabled: true\n")
    (profile / "memories").mkdir()
    memory_path = profile / "memories/MEMORY.md"
    memory_path.write_text("Existing profile memory must stay unchanged.")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(server, "_hermes_home", root)
    monkeypatch.setattr(server, "_sessions", {"s": {"agent": None, "profile_home": str(profile),
                                                   "source": "desktop", "running": False, "cwd": str(tmp_path)}})
    monkeypatch.setattr(server, "_pending_subcommand_store", lambda *_: (_ for _ in ()).throw(AssertionError("Status loaded a mutable store")))
    before = {p: p.read_bytes() for p in (root / "config.yaml", profile / "config.yaml", memory_path)}
    response = server._methods["command.dispatch"]("r", {"name": "memory", "arg": "status --json", "session_id": "s"})
    assert "error" not in response, response
    data = json.loads(response["result"]["output"])
    assert data["profile_home"] == str(profile)
    assert data["configuration"]["background_review_enabled"] is True
    assert {p: p.read_bytes() for p in before} == before
    assert not (profile / "memories/USER.md").exists()
