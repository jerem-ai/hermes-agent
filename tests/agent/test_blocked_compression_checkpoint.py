"""Oversized blocked recovery preserves the task without another provider call."""
from unittest.mock import patch
import pytest
from tests.agent import test_completion_announcement_recovery as completion_helpers

guard_agent = completion_helpers.guard_agent
from agent.turn_preflight_gate import run_preflight_gate


def gate(agent, pressure):
    agent.compression_enabled = True
    compressor = agent.context_compressor
    compressor.context_length = 65536
    compressor.threshold_tokens = 55705
    compressor._record_structural_no_op('fixture has no compressible middle')
    compressor.should_defer_preflight_to_real_usage = lambda count: False
    messages = [
        {'role': 'system', 'content': 'Keep the existing resource limits.'},
        {'role': 'user', 'content': 'Repair /workspace/service.py and run its tests.'},
        {'role': 'assistant', 'content': 'I will inspect the remaining caller path.', 'tool_calls': [{'id': 'source-read', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{"path":"/workspace/service.py"}'}}]},
        {'role': 'tool', 'tool_call_id': 'source-read', 'name': 'read_file', 'content': 'source evidence\n' * 16000},
    ]
    with patch('agent.turn_preflight.ensure_compression_feasibility_checked'):
        return run_preflight_gate(agent, request_pressure_tokens=pressure,
            _moa_prepared_request=None, pending_moa_prepared_request=None, messages=messages,
            system_message=None, user_message=messages[1]['content'], active_system_prompt=messages[0]['content'],
            conversation_history=None, api_call_count=1, compression_attempts=0, max_compression_attempts=3,
            effective_task_id='fixture', final_response=None, failed=False, _turn_exit_reason=None,
            _compression_timeout_exhausted=False, _preflight_compression_blocked=False,
            _provider_overflow_recovery_pending=False, _last_preflight_pressure=None)


def test_oversized_structurally_blocked_request_saves_pending_task(guard_agent):
    agent, db = guard_agent
    verdict = gate(agent, 66195)
    assert verdict.action == 'return'
    result = verdict.result
    assert result['completed'] is False
    assert result['checkpoint_saved'] is True
    assert result['compression_deferred'] is True
    assert not result.get('compression_exhausted')
    assert 'service.py' in result['checkpoint']['goal']
    assert 'caller path' in result['checkpoint']['pending_step']
    assert agent.client.chat.completions.create.call_count == 0
    assert any('service.py' in str(m.get('content')) for m in db.get_messages_as_conversation(agent.session_id))


def test_large_but_fitting_request_can_continue_during_structural_backoff(guard_agent):
    agent, _db = guard_agent
    assert gate(agent, 62423).action == 'fallthrough'


def test_checkpoint_write_failure_is_reported_without_saved_claim(guard_agent, monkeypatch):
    agent, _db = guard_agent
    monkeypatch.setattr(agent, '_persist_session', lambda *args: False)
    result = gate(agent, 66195).result
    assert result['checkpoint_saved'] is False
    assert result['failed'] is True
    assert 'could not save' in result['final_response'].lower()


@pytest.mark.parametrize("control", ["interrupt", "redirect", "steer"])
def test_pending_user_control_prevents_stale_checkpoint(guard_agent, control):
    agent, _db = guard_agent
    with patch('agent.turn_final_response._queued_user_control', return_value=control), patch.object(agent, '_persist_session') as save:
        verdict = gate(agent, 66195)
    assert verdict.action == 'continue'
    assert verdict.api_call_count == 0
    save.assert_not_called()
    assert agent.client.chat.completions.create.call_count == 0
