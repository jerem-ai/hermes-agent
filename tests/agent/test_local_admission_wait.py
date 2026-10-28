"""A local admission refusal keeps the exact pending request and supports Stop/steer."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.turn_local_admission import local_admission_pending, wait_for_local_admission
from agent.turn_retry_state import TurnRetryState
from tests.agent import test_outer_error_retry_bound

recovered_agent = test_outer_error_retry_bound.loop_agent


def error(*reasons):
    return SimpleNamespace(status_code=502, body={"error": {
        "code": "all_routes_failed", "failures": [{"reason": reason} for reason in reasons],
    }})


def agent():
    a = MagicMock()
    a.base_url = "http://127.0.0.1:8790/v1"
    a.provider, a.model, a.api_key, a.platform = "custom", "local/model", "test", "desktop"
    a.context_compressor = None
    a._current_streamed_assistant_text = ""
    a._has_content_after_think_block.side_effect = lambda text: bool(text.strip())
    return a


def test_busy_waits_outlast_retry_ladder_without_mutating_prior_tool_results(monkeypatch):
    from agent.turn_api_error import handle_api_error
    a, retry = agent(), TurnRetryState()
    history = [{"role": "user", "content": "Continue from the saved work."}]
    for index in range(27):
        call_id = f"already-done-{index}"
        history.extend([
            {"role": "assistant", "tool_calls": [{"id": call_id}]},
            {"role": "tool", "tool_call_id": call_id, "content": "chunk " * 2048 + "Crème 雪 0"},
        ])
    assert len(history) == 55 and sum(len(row.get("content", "")) for row in history) > 330000
    original = repr(history)
    waits = []
    monkeypatch.setattr("agent.turn_recovery.interruptible_backoff_sleep", lambda *args, **kw: waits.append(kw))
    monkeypatch.setattr("agent.turn_api_error.recover_before_classification", lambda *a, **kw: (False, "stable system"))
    a._extract_api_error_context.return_value = {}
    for _ in range(8):
        verdict = handle_api_error(a, api_error=error("local_busy"), _retry=retry,
            thinking_spinner=None, messages=history, api_messages=history, api_kwargs={},
            system_message="stable system", active_system_prompt="stable system",
            conversation_history=history, approx_tokens=83000, retry_count=0, max_retries=3,
            compression_attempts=0, max_compression_attempts=3, api_call_count=22,
            api_request_id="pending-step", api_start_time=0, effective_task_id="multi-step", turn_id="turn")
        assert verdict.action == "continue" and verdict.retry_count == 0
        assert verdict.messages is history
    assert repr(history) == original
    assert retry.auto_recovery_cycles_used == 0
    assert len(waits) == 8 and all(wait["messages"] is history for wait in waits)
    assert "Waiting for the local model" in a._emit_diagnostic_wait.call_args.args[0]
    assert wait_for_local_admission(a, error("upstream_5xx"), retry,
        messages=history, conversation_history=history, api_call_count=22) is None


@pytest.mark.parametrize("reasons", [("http_423",), ("local_locked",), ("auth_error",),
    ("local_busy", "upstream_5xx"), ()])
def test_safety_locks_unknown_errors_and_mixed_failures_do_not_wait(reasons):
    assert not local_admission_pending(agent(), error(*reasons))


def test_remote_gateway_and_partially_delivered_response_do_not_replay():
    a = agent()
    a.base_url = "https://provider.example/v1"
    assert not local_admission_pending(a, error("local_busy"))
    a.base_url = "http://[::1]:8790/v1"
    assert local_admission_pending(a, error("local_busy"))
    a._current_streamed_assistant_text = "Delivered answer"
    assert not local_admission_pending(a, error("local_busy"))


def test_streaming_does_not_issue_a_second_probe_for_known_admission_refusal():
    from tests.agent.test_stream_unmask_5xx import _make_call

    call = _make_call({"model": "local/model", "stream": True, "messages": []})
    call.agent.base_url = "http://127.0.0.1:8790/v1"
    call.agent._has_content_after_think_block = lambda text: bool(text.strip())
    assert not call._unmask_server_error_with_nonstreaming(error("local_busy"))
    assert call.agent._stream_5xx_probe_ts is None


def test_older_runtime_uses_its_native_status_and_thinking_callbacks(monkeypatch):
    legacy = SimpleNamespace(base_url="http://localhost:8790/v1", platform="cli",
        _current_streamed_assistant_text="", _has_content_after_think_block=bool,
        _vprint=MagicMock(), thinking_callback=MagicMock(), _touch_activity=MagicMock())
    monkeypatch.setattr("agent.turn_recovery.interruptible_backoff_sleep", lambda *args, **kw: None)
    assert wait_for_local_admission(legacy, error("local_busy"), TurnRetryState(),
        messages=[], conversation_history=[], api_call_count=1) == {"action": "continue"}
    assert legacy._vprint.call_args.kwargs == {"force": True}
    assert legacy.thinking_callback.call_args.args == legacy._vprint.call_args.args


@pytest.mark.parametrize("outcome,redirect,action", [({"interrupted": True}, False, "return"), (None, True, "break")])
def test_stop_and_steer_use_native_interrupt_path(monkeypatch, outcome, redirect, action):
    retry = TurnRetryState(restart_with_redirected_messages=redirect)
    monkeypatch.setattr("agent.turn_recovery.interruptible_backoff_sleep", lambda *a, **kw: outcome)
    result = wait_for_local_admission(agent(), error("local_busy"), retry,
        messages=[], conversation_history=[], api_call_count=1)
    assert result["action"] == action
    if outcome is not None:
        assert result["result"] is outcome


def test_real_turn_resumes_after_more_busy_refusals_than_recovery_cycles(recovered_agent, monkeypatch):
    import httpx
    from openai import InternalServerError
    from tests.agent.test_run_agent import _mock_response

    a = recovered_agent
    a.base_url = "http://127.0.0.1:8790/v1"
    a._disable_streaming = True
    response = httpx.Response(502, request=httpx.Request("POST", a.base_url))
    refusal = InternalServerError("not admitted", response=response, body=error("local_busy").body)
    a.client.chat.completions.create.side_effect = [refusal] * 60 + [_mock_response(content="Resumed", finish_reason="stop")]
    monkeypatch.setattr("agent.turn_recovery.interruptible_backoff_sleep", lambda *args, **kw: None)
    result = a.run_conversation("Continue from the saved step.")
    assert result["completed"] is True
    assert result["final_response"] == "Resumed"
    assert a.client.chat.completions.create.call_count == 61
