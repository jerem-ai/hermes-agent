"""A malformed provider scalar must not break a healthy checkpoint store."""
import pytest
from tests.agent import test_completion_announcement_recovery as completion_helpers
from tests.agent.test_completion_announcement_recovery import _run, _final, _tool

guard_agent = completion_helpers.guard_agent


@pytest.mark.parametrize("stage", ["provider", "output_hook"])
def test_checkpoint_keeps_canonical_unicode_safety(guard_agent, stage, monkeypatch):
    agent, db = guard_agent
    action = "Let me now inspect the remaining source."
    if stage == "provider":
        action = "Let me now inspect \ud800source."
    else:
        def hook(name, *args, **kwargs):
            return ["\ud800" + kwargs["response_text"]] if name == "transform_llm_output" else []
        monkeypatch.setattr("agent.turn_finalizer._invoke_hook_safely", hook)
    result, effects = _run(agent, "Finish the approved repair.", [_tool(), _final(action), _final(action), _final(action)])
    assert effects == 1
    assert result["completed"] is False
    assert result["checkpoint_saved"] is True
    assert result["failure_reason"] == "loop_error"
    result["final_response"].encode("utf-8")
    for message in result["messages"]:
        if isinstance(message.get("content"), str):
            message["content"].encode("utf-8")
    assert any(m.get("content") == result["final_response"] for m in db.get_messages_as_conversation(agent.session_id))
