"""Desktop consumes native failed turns as errors and completed clarifications as answers."""
import pytest

from tests.agent import test_completion_announcement_recovery as helpers
from tests.agent.test_completion_announcement_recovery import _run, _final, _tool
from tui_gateway.prompt_turn import _turn_outcome

guard_agent = helpers.guard_agent


@pytest.mark.parametrize("stalls,text", [
    (True, "Let me add the selected entry to /tmp/雪/config:"),
    (False, "Let me add one clarification: access already works."),
    (True, "Give me a minute to check what's available..."),
    (False, 'The subtask is complete. The log said "give me a minute to check..."'),
])
def test_desktop_status_matches_native_file_action_result(guard_agent, stalls, text):
    agent, db = guard_agent
    stages = [_tool()] + [_final(text)] * (3 if stalls else 1)
    result, effects = _run(agent, "Finish the authorized setup.", stages)
    raw, status, _ = _turn_outcome(result)
    assert effects == 1
    assert status == ("error" if stalls else "complete")
    assert result["completed"] is not stalls
    assert db.get_messages_as_conversation(agent.session_id)[-1]["content"] == raw
    if stalls:
        assert "Pending step:" in raw
    else:
        assert raw == text
