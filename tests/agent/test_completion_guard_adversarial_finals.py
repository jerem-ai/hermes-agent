"""Guard-only action detection and steer arrival through the public delivery callback."""
import pytest
from tests.agent import test_completion_announcement_recovery as completion_helpers
from tests.agent.test_completion_announcement_recovery import _run, _final, _tool, DONE

guard_agent = completion_helpers.guard_agent


@pytest.mark.parametrize("after_tool", [False, True])
@pytest.mark.parametrize("text", ["Let me be clear: all checks passed.", "Done. Let me say thank you."])
def test_completed_performative_stays_final(guard_agent, after_tool, text):
    agent, _ = guard_agent
    stages = ([_tool()] if after_tool else []) + [_final(text), _final(DONE)]
    result, effects = _run(agent, "Report the completed checks.", stages)
    assert result["final_response"] == text
    assert result["completed"] is True
    assert agent.client.chat.completions.create.call_count == 1 + int(after_tool)
    assert effects == int(after_tool)


@pytest.mark.parametrize("historical_tool", [False, True])
def test_steer_during_nudge_is_in_next_provider_request(guard_agent, historical_tool):
    from agent.completion_announcements import latest_execution_request_text
    agent, _ = guard_agent
    queued = []

    def user_intervenes(text, **kwargs):
        if not queued:
            queued.append(agent.steer("Pause and wait for my approval."))

    agent.interim_assistant_callback = user_intervenes

    def next_response():
        request = agent.client.chat.completions.create.call_args.kwargs
        latest = latest_execution_request_text(request["messages"])
        return _final("I will now wait for your approval.") if "Pause and wait" in latest else _tool()

    history = None
    if historical_tool:
        history = [{"role": "user", "content": "Old completed task."},
                   {"role": "assistant", "content": "", "tool_calls": [{"id": "historic_r", "type": "function", "function": {"name": "web_search", "arguments": "{}"}}]},
                   {"role": "tool", "tool_call_id": "historic_r", "content": "Old completed evidence."},
                   {"role": "assistant", "content": "The old task is done."}]
    result, effects = _run(agent, "Finish the approved repair.", [_final("Let me now inspect the source."), next_response, _final("I will now wait for your approval.")], history)
    assert queued == [True]
    assert effects == 0
    assert agent.client.chat.completions.create.call_count == 2
    assert result["final_response"] == "I will now wait for your approval."
