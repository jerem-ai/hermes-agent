"""The native memory-status action runs through a real persisted turn."""
import json

import pytest

from tests.agent.test_completion_announcement_recovery import _final, _run
from tests.agent.test_run_agent import _mock_response, _mock_tool_call
from tools.memory_tool import MEMORY_SCHEMA

pytest_plugins = ["tests.agent.test_completion_announcement_recovery"]


def test_status_turn_preserves_prompt_tools_and_saved_caller_result(guard_agent):
    agent, db = guard_agent
    agent.tools = [{"type": "function", "function": MEMORY_SCHEMA}]
    agent.valid_tool_names = {"memory"}
    history = []
    for i in range(25):
        history += [{"role": "user", "content": f"Prior task {i}: quoted 雪 " * 75},
                    {"role": "assistant", "content": "Recorded fixture context. " * 75}]
    response = _mock_response(content="", finish_reason="tool_calls", tool_calls=[
        _mock_tool_call(name="memory", arguments='{"action":"status"}', call_id="learning-status")])
    answer = "Automatic review is configured. No completed review is recorded in this evidence."
    result, effects = _run(agent, "Inspect your learning status without changing anything.",
                           [response, _final(answer)], history=history)
    assert effects == 0  # memory is an inline tool, not a mocked external dispatch.
    assert result["completed"] and result["final_response"] == answer
    tool = next(m for m in result["messages"] if m.get("tool_call_id") == "learning-status" and m["role"] == "tool")
    assert json.loads(tool["content"])["observed"]["outcome_evidence"] == "not_recorded"
    requests = agent.client.chat.completions.create.call_args_list
    assert requests[0].kwargs["tools"] == requests[1].kwargs["tools"]
    assert [m for m in requests[0].kwargs["messages"] if m["role"] == "system"] == [
        m for m in requests[1].kwargs["messages"] if m["role"] == "system"]
    saved = db.get_messages_as_conversation(agent.session_id)
    assert saved[-1]["content"] == answer
