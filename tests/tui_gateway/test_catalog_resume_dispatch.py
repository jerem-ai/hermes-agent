"""A cold slash catalog must not hold the socket's reconnect RPCs behind plugin discovery."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import pytest

from hermes_state import SessionDB
from tui_gateway import server, ws as ws_mod


class Socket:
    def __init__(self):
        self.incoming = asyncio.Queue()
        self.outgoing = asyncio.Queue()
        self.received = {}

    async def accept(self):
        pass

    async def send_text(self, text):
        await self.outgoing.put(json.loads(text))

    async def receive_text(self):
        frame = await self.incoming.get()
        if frame is None:
            raise ws_mod._WebSocketDisconnect()
        return json.dumps(frame)

    async def close(self, **kwargs):
        await self.incoming.put(None)

    async def request(self, rid, method, **params):
        await self.incoming.put({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})

    async def reply(self, rid):
        if rid in self.received:
            return self.received.pop(rid)
        while True:
            frame = await self.outgoing.get()
            if frame.get("id") == rid:
                return frame
            if "id" in frame:
                self.received[frame["id"]] = frame


def mount(monkeypatch, tmp_path):
    home = tmp_path / ".hermes" / "profiles" / "texting"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text("model:\n  default: fixture-model\n  provider: custom\n")
    (home / ".env").write_text("")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(server, "_hermes_home", str(home))
    db = SessionDB(home / "state.db")
    db.create_session("saved-chat", source="desktop", model="fixture-model")
    # About 216k characters of persisted context, with historical tool-call boundaries.
    for index in range(52):
        db.append_message("saved-chat", "user", f"request {index}: " + "old context " * 160)
        if index % 13 == 0:
            call_id = f"read-{index}"
            db.append_message("saved-chat", "assistant", "Checking the stored artifact", tool_calls=[{
                "id": call_id, "type": "function", "function": {
                    "name": "read_file", "arguments": json.dumps({"path": f"/repo/artifact-{index}.json"})}}])
            db.append_message("saved-chat", "tool", '{"verified":true}', tool_call_id=call_id)
        db.append_message("saved-chat", "assistant", "stored answer " * 160)
    other = home.parent / "other"
    other.mkdir()
    (other / "config.yaml").write_text("model:\n  default: other-model\n  provider: custom\n")
    (other / ".env").write_text("")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_default_session_cwd", lambda: str(tmp_path))
    monkeypatch.setattr(server, "_start_agent_build", lambda *args: None)
    monkeypatch.setattr(server, "_maybe_schedule_auto_continue", lambda *args: None)
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    for name in ("_ensure_skin_watcher", "_ensure_lease_watcher", "_start_backend_heartbeat_refresher",
                 "_schedule_startup_orphan_sweep"):
        monkeypatch.setattr(server, name, lambda: None)
    monkeypatch.setattr(ws_mod, "_note_dashboard_client_activity", lambda **kwargs: None)
    monkeypatch.setattr(server, "resolve_skin", lambda: "default")
    from tui_gateway import command_discovery
    monkeypatch.setattr(command_discovery, "_catalog_skills", lambda cat, skills, module_loader=None: "")
    return db


@pytest.mark.parametrize("plugin_failure", [False, True], ids=["unusual-command", "discovery-failure"])
def test_cold_catalog_does_not_block_reconnect_on_the_same_socket(monkeypatch, tmp_path, plugin_failure):
    from hermes_cli import plugins

    db = mount(monkeypatch, tmp_path)
    started, release = threading.Event(), threading.Event()
    before = db.get_messages_as_conversation("saved-chat", repair_alternation=False)
    existing = set(server._sessions)
    seen_profiles = []

    def discover():
        from hermes_constants import get_hermes_home
        profile = get_hermes_home().name
        seen_profiles.append(profile)
        started.set()
        assert release.wait(10), "test must release the simulated plugin startup"
        if plugin_failure and profile == "texting":
            raise RuntimeError("fixture plugin failed to register")
        name = "odd-name" if profile == "texting" else "other-only"
        return {name: {"description": "line one\n日本語", "argument_mode": "mixed"}}

    monkeypatch.setattr(plugins, "get_plugin_commands", discover)

    async def run():
        socket = Socket()
        connection = asyncio.create_task(ws_mod.handle_ws(socket))
        try:
            await socket.request("catalog", "commands.catalog", profile="texting")
            assert await asyncio.to_thread(started.wait, 3), "catalog never reached plugin discovery"
            await socket.request("other-catalog", "commands.catalog", profile="other")
            # Cached runtime IDs from the previous backend are gone. Their rejection and the
            # durable resume must be admitted while the cosmetic command warm-up is still blocked.
            await socket.request("stale", "session.activate", session_id="previous-backend-runtime")
            await socket.request("resume", "session.resume", session_id="saved-chat", source="desktop",
                                 profile="texting", defer_history=True, omit_messages=True)
            stale = await asyncio.wait_for(socket.reply("stale"), 3)
            assert stale["error"]["code"] == 4001
            resumed = await asyncio.wait_for(socket.reply("resume"), 3)
            assert "error" not in resumed, resumed
            result = resumed["result"]
            assert result["resumed"] == "saved-chat"
            assert result["session_id"] != "previous-backend-runtime"
            assert result["message_count"] == 112
            assert result["messages"] == []
            assert not release.is_set(), "resume must not require plugin discovery to finish"
            release.set()
            catalog = await asyncio.wait_for(socket.reply("catalog"), 3)
            assert "error" not in catalog, catalog
            if plugin_failure:
                assert "fixture plugin failed" in catalog["result"]["warning"]
            else:
                assert catalog["result"]["commands"]["/odd-name"]["argument_mode"] == "mixed"
                assert ["/odd-name", "line one\n日本語"] in catalog["result"]["pairs"]
            other_catalog = await asyncio.wait_for(socket.reply("other-catalog"), 3)
            assert "/other-only" in other_catalog["result"]["commands"]
            assert "/odd-name" not in other_catalog["result"]["commands"]
            assert "/other-only" not in catalog["result"]["commands"]
            assert sorted(seen_profiles) == ["other", "texting"]
        finally:
            release.set()
            await socket.incoming.put(None)
            await asyncio.wait_for(connection, 5)

    try:
        asyncio.run(run())
        assert db.get_messages_as_conversation("saved-chat", repair_alternation=False) == before
    finally:
        release.set()
        for sid in set(server._sessions) - existing:
            server._sessions.pop(sid, None)
        db.close()
