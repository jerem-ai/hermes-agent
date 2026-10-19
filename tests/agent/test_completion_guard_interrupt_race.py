"""Cancellation arriving at the final guard's control read still belongs to the loop."""
from tests.agent import test_completion_announcement_recovery as completion_helpers
from tests.agent.test_completion_announcement_recovery import _run, _final

guard_agent = completion_helpers.guard_agent


def test_cancel_after_provider_response_before_guard_decision(guard_agent, monkeypatch):
    import agent.turn_final_response as phase
    agent, _ = guard_agent
    from unittest.mock import MagicMock
    callback = MagicMock()
    agent.interim_assistant_callback = callback
    original = phase._queued_user_control

    def cancel_then_read(owner):
        owner._interrupt_requested = True
        return original(owner)

    # Inject only the arrival timing; execute the actual user-control reader and turn loop.
    monkeypatch.setattr(phase, "_queued_user_control", cancel_then_read)
    result, effects = _run(agent, "Finish the approved repair.", [_final("Let me now inspect the source.")])
    assert agent.client.chat.completions.create.call_count == 1
    assert effects == 0
    callback.assert_not_called()
    assert result["completed"] is False
    assert result["interrupted"] is True
