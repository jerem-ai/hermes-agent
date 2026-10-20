"""Browse and bad anchors must offer a usable next call without inventing message IDs."""
import json
from pathlib import Path

from hermes_state import SessionDB
from tools.session_search_tool import session_search


def test_browse_and_wrong_count_anchor_recover_with_real_ids(tmp_path):
    with SessionDB(tmp_path / "state.db") as db:
        db.create_session("unrelated", source="cli")
        for _ in range(40):
            db.append_message("unrelated", role="user", content="Earlier unrelated message.")
        sid = "desktop-雪"
        db.create_session(sid, source="desktop")
        db.append_message(sid, role="user", content="Inspect the saved config.")
        db.append_message(sid, role="assistant", content="quoted result " * 20000)
        last_id = db.append_message(sid, role="assistant", content="Existing SSH access works.")
        browse = json.loads(session_search(db=db, current_session_id="unrelated"))
        assert "session_id alone" in browse["message"]
        wrong = json.loads(session_search(db=db, session_id=sid, around_message_id=3))
        assert wrong["success"] is False
        assert wrong["recovery_call"] == {"session_id": sid}
        recovered = json.loads(session_search(db=db, **wrong["recovery_call"]))
        assert recovered["success"] is True
        assert recovered["messages"][-1]["id"] == last_id
        assert recovered["messages"][-1]["content"] == "Existing SSH access works."
        scrolled = json.loads(session_search(db=db, session_id=sid, around_message_id=last_id))
        assert scrolled["success"] is True
        assert any(m.get("anchor") and m["id"] == last_id for m in scrolled["messages"])


def test_recovery_call_keeps_the_explicit_profile(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    other = root / "profiles/work"
    other.mkdir(parents=True)
    (other / "config.yaml").write_text("model: {}\n")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    with SessionDB(other / "state.db") as remote, SessionDB(root / "state.db") as local:
        remote.create_session("only-work", source="desktop")
        remote.append_message("only-work", role="user", content="A scoped session.")
        failure = json.loads(session_search(db=local, profile="work", session_id="only-work", around_message_id=999))
        assert failure["success"] is False
        assert failure["recovery_call"] == {"session_id": "only-work", "profile": "work"}
        accepted = json.loads(session_search(db=local, **failure["recovery_call"]))
        assert accepted["success"] is True
        assert accepted["messages"][0]["content"] == "A scoped session."
        assert json.loads(session_search(db=local, session_id="only-work"))["success"] is False
