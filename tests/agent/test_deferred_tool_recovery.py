"""A hidden tool is discoverable; rejection must not claim the capability is absent."""
import json

from agent.conversation_loop import _invalid_tool_name_error_content
from tests.agent.test_completion_announcement_recovery import _final, _run, _tool
from tests.agent.test_run_agent import _mock_response, _mock_tool_call
from tools.session_search_tool import SESSION_SEARCH_SCHEMA

pytest_plugins = ["tests.agent.test_completion_announcement_recovery"]


def test_unknown_direct_call_points_to_discovery_without_claiming_absence(guard_agent):
    agent, db = guard_agent
    agent.valid_tool_names.add("tool_search")
    # Capture the actual rejected tool result sent on the next full-turn API request.
    result, effects = _run(agent, "Can you search yesterday's conversation?", [
        _tool(name="session_search"), _final("The tool is deferred. No history was read in this turn."),
    ])
    requests = agent.client.chat.completions.create.call_args_list
    rejection = next(m["content"] for m in requests[1].kwargs["messages"] if m.get("role") == "tool")
    assert "does not exist" not in rejection
    assert "tool_search" in rejection and "session_search" in rejection
    assert effects == 0
    assert result["completed"] is True
    assert rejection in [m.get("content") for m in db.get_messages_as_conversation(agent.session_id)]
    assert requests[0].kwargs["tools"] == requests[1].kwargs["tools"]


def test_direct_history_search_recovers_prior_session_through_real_inline_executor(guard_agent):
    agent, db = guard_agent
    db.create_session("prior-evening", source="bluebubbles")
    db.append_message("prior-evening", role="assistant", content="Committed fixture123 in the MacBook workspace.")
    agent.valid_tool_names.add("session_search")
    agent.tools.append({"type": "function", "function": SESSION_SEARCH_SCHEMA})
    response = _mock_response(content="", finish_reason="tool_calls", tool_calls=[
        _mock_tool_call(name="session_search", arguments=json.dumps({"session_id": "prior-evening"}),
                        call_id="recall-prior"),
    ])
    result, effects = _run(agent, "Check what you committed yesterday.", [
        response, _final("The prior session records fixture123. Its repository still needs verification."),
    ])
    tool_result = next(m["content"] for m in result["messages"] if m.get("role") == "tool")
    assert "fixture123" in tool_result and "MacBook" in tool_result
    assert effects == 0  # The inline executor read the temp DB; no mocked external tool ran.
    assert result["completed"] is True
    assert any(m.get("content") == result["final_response"]
               for m in db.get_messages_as_conversation(agent.session_id))
    requests = agent.client.chat.completions.create.call_args_list
    assert requests[0].kwargs["tools"] == requests[1].kwargs["tools"]


def test_blank_tool_keeps_terse_anti_echo_error():
    result = _invalid_tool_name_error_content("", {"tool_search", "terminal"})
    assert "empty" in result and "Available tools" not in result


def test_surface_without_discovery_does_not_offer_it():
    result = _invalid_tool_name_error_content("nonexistent_工具", {"terminal"})
    assert "tool_search" not in result
    assert "terminal" in result
