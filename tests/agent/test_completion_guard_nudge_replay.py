"""Both guard nudges retain their internal-user status after real DB persistence."""
import pytest
from tests.agent import test_completion_announcement_recovery as completion_helpers
from tests.agent.test_completion_announcement_recovery import _run, _final, _tool, DONE

guard_agent = completion_helpers.guard_agent


@pytest.mark.parametrize("kind", ["ack", "degenerate"])
def test_guard_nudge_survives_db_and_rebuilt_context(guard_agent, kind):
    from agent.conversation_loop import _CODEX_ACK_CONTINUATION_NUDGE, _DEGENERATE_FINAL_NUDGE
    from agent.context_compressor import ContextCompressor, SUMMARY_PREFIX
    from agent.completion_announcements import latest_execution_request_text
    agent, db = guard_agent
    user = "Finish the approved repair."
    if kind == "degenerate":
        agent._intent_ack_continuation = True
    candidate = "?oops" if kind == "degenerate" else "Let me now inspect the source."
    expected = _DEGENERATE_FINAL_NUDGE if kind == "degenerate" else _CODEX_ACK_CONTINUATION_NUDGE
    result, effects = _run(agent, user, [_tool(), _final(candidate), _final(DONE)])
    assert result["completed"] is True
    assert effects == 1
    plain = [dict(m) for m in db.get_messages_as_conversation(agent.session_id)]
    nudge = next(m for m in plain if m["role"] == "user" and m["content"] == expected)
    assert ContextCompressor._is_synthetic_compression_user_turn({"role": "user", "content": nudge["content"]}) is True
    assert latest_execution_request_text(plain) == user
    rebuilt = [{"role": "user", "content": SUMMARY_PREFIX + "\nOlder context summarized."},
               {"role": "assistant", "content": "Historical context received."}] + plain
    assert latest_execution_request_text(rebuilt) == user
    assert ContextCompressor._is_synthetic_compression_user_turn(nudge) is True
