"""The texting profile can keep history eager without changing other profiles' tool grants."""
import json

from hermes_state import SessionDB
from tools.session_search_tool import SESSION_SEARCH_SCHEMA, session_search
from tools.tool_search import ToolSearchConfig, assemble_tool_defs


def test_eager_texting_history_preserves_deferred_tools_and_profile_isolation(tmp_path):
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    defs = [{"type": "function", "function": SESSION_SEARCH_SCHEMA}, {
        "type": "function", "function": {"name": "computer_use", "description": "Fixture",
                                           "parameters": {"type": "object", "properties": {}}},
    }]
    default_defer = DEFAULT_CONFIG["tools"]["tool_search"]["defer"]
    texting_defer = [name for name in default_defer if name != "session_search"]
    names = lambda result: {td["function"]["name"] for td in result.tool_defs}
    texting = assemble_tool_defs(defs, context_length=65536,
                                 config=ToolSearchConfig.from_raw({"defer": texting_defer}))
    other = assemble_tool_defs(defs, context_length=65536,
                               config=ToolSearchConfig.from_raw({"defer": default_defer}))
    assert "session_search" in names(texting) and "computer_use" not in names(texting)
    assert "session_search" not in names(other) and "tool_call" in names(other)
    # Search executes against the caller's injected DB; it needs no auxiliary model.
    with SessionDB(tmp_path / "texting.db") as texting_db, SessionDB(tmp_path / "other.db") as other_db:
        for db, session, text in [(texting_db, "prior-texting", "Committed fixture123 in the remote workspace."),
                                  (other_db, "other-profile", "Private unrelated record.")]:
            db.create_session(session, source="bluebubbles")
            db.append_message(session, role="assistant", content=text)
        found = json.loads(session_search(query="fixture123", db=texting_db))
        missed = json.loads(session_search(query="fixture123", db=other_db))
        assert "fixture123" in json.dumps(found) and found["success"]
        assert "fixture123" not in json.dumps(missed.get("results", []))
