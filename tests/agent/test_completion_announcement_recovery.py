"""Full-turn contracts for premature completion, using the real tool executor and temp DB."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import sqlite3

import pytest

from hermes_state import SessionDB
from tests.agent.test_run_agent import _mock_response, _mock_tool_call


CASES = json.loads((Path(__file__).parent / "fixtures/completion_announcements.json").read_text())["cases"]
CASES += json.loads((Path(__file__).parent / "fixtures/completion_holdout.json").read_text())["cases"]
CASES += json.loads((Path(__file__).parent / "fixtures/completion_holdout_v2.json").read_text())["cases"]
CASES += json.loads((Path(__file__).parent / "fixtures/completion_holdout_v3.json").read_text())["cases"]
CASES += json.loads((Path(__file__).parent / "fixtures/completion_holdout_v4.json").read_text())["cases"]
DONE = "The requested repair is complete and its checks passed."


@pytest.fixture
def guard_agent(tmp_path, monkeypatch):
    from run_agent import AIAgent

    monkeypatch.setattr("agent.title_generator.maybe_auto_title", lambda *a, **k: None)
    defs = [{"type": "function", "function": {"name": "web_search", "description": "Fixture read", "parameters": {"type": "object", "properties": {}}}}]
    with (
        patch("model_tools.get_tool_definitions", return_value=defs),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.model_metadata.fetch_model_metadata", return_value={}),
        patch("agent.model_metadata.get_model_context_length", return_value=65536),
        patch("hermes_cli.config.load_config_readonly", return_value={
            "model": {"default": "leyline/tars", "provider": "custom", "base_url": "http://127.0.0.1:1/v1", "context_length": 65536},
            "compression": {"enabled": True, "threshold": 0.8, "target_ratio": 0.2, "protect_first_n": 3, "protect_last_n": 20},
        }),
    ):
        db = SessionDB(db_path=tmp_path / "state.db")
        agent = AIAgent(
            api_key="fixture-key", base_url="http://127.0.0.1:1/v1",
            provider="custom", api_mode="chat_completions", model="leyline/tars",
            max_iterations=12, quiet_mode=True, skip_context_files=True,
            skip_memory=True, skip_background_review=True, session_db=db,
            checkpoints_enabled=False, max_tokens=4096,
        )
    assert agent._config_context_length == 65536
    assert agent.context_compressor.context_length == 65536
    agent.client = MagicMock()
    agent._cached_system_prompt = "Complete only the authorized task."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent._intent_ack_continuation = "auto"
    agent._stall_guards = True
    agent._api_max_retries = 1
    agent._auto_recovery_cycles = 0
    agent.tool_delay = 0
    yield agent, db
    db.close()


def _final(text):
    return _mock_response(content=text, finish_reason="stop")


def _tool(name="web_search", call_id="read_1"):
    return _mock_response(content="", finish_reason="tool_calls", tool_calls=[
        _mock_tool_call(name=name, arguments="{}", call_id=call_id),
    ])


def _run(agent, user, stages, history=None, tool_result='{"results": "fixture evidence"}'):
    remaining = iter(stages)
    def respond(*args, **kwargs):
        stage = next(remaining)
        if isinstance(stage, BaseException):
            raise stage
        return stage() if callable(stage) else stage
    agent.client.chat.completions.create.side_effect = respond
    with (
        patch("model_tools.handle_function_call", return_value=tool_result) as execute,
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation(user, conversation_history=history)
    return result, execute.call_count


@pytest.mark.parametrize("after_tool", [False, True], ids=["initial", "post-tool"])
@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_announcement_contract_on_texting_transport(guard_agent, case, after_tool):
    agent, db = guard_agent
    stages = ([_tool()] if after_tool else []) + [_final(case["assistant"])]
    stages += ([_tool(call_id="continued_1"), _final(DONE)] if case["continue"] else [_final(DONE)])
    result, effects = _run(agent, case["user"], stages)
    assert agent.api_mode == "chat_completions"
    assert result["completed"] is True
    assert result["final_response"] == (DONE if case["continue"] else case["assistant"])
    assert agent.client.chat.completions.create.call_count == int(after_tool) + 1 + 2 * int(case["continue"])
    assert effects == int(after_tool) + int(case["continue"])
    requests = [call.kwargs for call in agent.client.chat.completions.create.call_args_list]
    assert all(r.get("tools") == requests[0].get("tools") for r in requests)
    systems = [[m for m in r["messages"] if m.get("role") == "system"] for r in requests]
    assert all(system == systems[0] for system in systems)
    durable = db.get_messages_as_conversation(agent.session_id)
    assert any(m.get("role") == "assistant" and m.get("content") == result["final_response"] for m in durable)


def test_exhaustion_is_bounded_non_success_with_durable_pending_step(guard_agent):
    agent, db = guard_agent
    action = "Let me now inspect the remaining file."
    result, effects = _run(agent, "Finish the approved repair.", [_tool(), _final(action), _final(action), _final(action)])
    assert agent.client.chat.completions.create.call_count == 4
    assert effects == 1
    assert result["completed"] is False
    assert result["failed"] is True
    assert result.get("failure_reason")
    assert result.get("checkpoint_saved") is True
    assert "could not be confirmed" not in result["final_response"]
    assert "inspect the remaining file" in result["final_response"].lower()
    durable = db.get_messages_as_conversation(agent.session_id)
    assert any(m.get("role") == "assistant" and m.get("content") == result["final_response"] for m in durable)
    from agent.completion_announcements import latest_execution_request_text
    from agent.context_compressor import SUMMARY_PREFIX, COMPRESSION_CONTINUATION_USER_CONTENT
    assert latest_execution_request_text(durable) == "Finish the approved repair."
    synthetic_tail = [{"role": "user", "content": SUMMARY_PREFIX + "\nEarlier repair summarized."},
                      {"role": "user", "content": COMPRESSION_CONTINUATION_USER_CONTENT}]
    assert latest_execution_request_text(durable + synthetic_tail) == "Finish the approved repair."
    resumed, repeated_effects = _run(agent, "Continue the approved repair.", [_final(action), _final(DONE)], durable)
    assert resumed["completed"] is True
    assert resumed["final_response"] == DONE
    assert agent.client.chat.completions.create.call_count == 6
    assert repeated_effects == 0


@pytest.mark.parametrize("control", ["disabled", "no_tools", "interrupt", "steer", "redirect", "checkpoint_disk_full", "checkpoint_unavailable", "checkpoint_raised", "checkpoint_transform", "mutating", "large_history", "counter_other_continue", "provider_error", "compression", "compression_interrupt", "safe_repeat", "uncertain_tool", "ack_off"])
def test_continuation_lifecycle_controls(guard_agent, control, monkeypatch):
    agent, db = guard_agent
    action = "Let me now inspect the remaining file."
    wait = "I will now wait for your approval."
    user = "Finish the approved repair."
    history = None
    tool_result = '{"results": "fixture evidence"}'
    compression_calls = []
    stages = [_final(action), _final(DONE)]
    expected_calls, expected_effects = 2, 0
    if control == "ack_off":
        from agent.agent_runtime_helpers import intent_ack_continuation_mode
        agent._intent_ack_continuation = False
        assert intent_ack_continuation_mode(agent) == "off"
    elif control == "disabled":
        agent._stall_guards = False
        expected_calls = 1
    elif control == "no_tools":
        agent.valid_tool_names = set()
        agent.tools = []
        expected_calls = 1
    elif control == "interrupt":
        def cancel():
            agent._interrupt_requested = True
            return _final(action)
        stages = [cancel]
        expected_calls = 1
    elif control in {"steer", "redirect"}:
        def correct():
            if control == "steer":
                assert agent.steer("Pause and wait for my approval.")
            else:
                assert agent.redirect("Pause and wait for my approval.")
            return _final(action)
        stages = [_tool(), correct, _final(wait)]
        expected_calls, expected_effects = 3, 1
    elif control.startswith("checkpoint_"):
        def fail_checkpoint():
            if control == "checkpoint_transform":
                def hooks(name, *args, **kwargs):
                    return ["[fixture transform] " + kwargs["response_text"]] if name == "transform_llm_output" else []
                monkeypatch.setattr("agent.turn_finalizer._invoke_hook_safely", hooks)
                monkeypatch.setattr(db, "append_messages_batch", MagicMock(side_effect=sqlite3.OperationalError("disk full")))
            elif control == "checkpoint_raised":
                monkeypatch.setattr(agent, "_persist_session", MagicMock(side_effect=OSError("fixture storage unavailable")))
            elif control == "checkpoint_disk_full":
                monkeypatch.setattr(db, "append_messages_batch", MagicMock(side_effect=sqlite3.OperationalError("disk full")))
            else:
                agent._persist_disabled = True
            return _final(action)
        stages = [_tool(), _final(action), _final(action), fail_checkpoint]
        expected_calls, expected_effects = 4, 1
    elif control == "mutating":
        agent.valid_tool_names = {"terminal"}
        agent.tools[0]["function"]["name"] = "terminal"
        stages = [_tool("terminal"), _final(action), _final(action), _final(action)]
        expected_calls, expected_effects = 4, 1
    elif control == "counter_other_continue":
        stages = [_tool(), _final(action), _mock_response(content="Working.", finish_reason="tool_calls"), _final(action), _final(action)]
        expected_calls, expected_effects = 5, 1
    elif control == "provider_error":
        class TransientProviderError(Exception):
            status_code = 500
        agent._api_max_retries = 2
        stages = [_tool(), _final(action), TransientProviderError("500 temporary fixture outage"), _final(action), _final(action)]
        expected_calls, expected_effects = 5, 1
    elif control in {"compression", "compression_interrupt"}:
        from agent.context_compressor import SUMMARY_PREFIX
        agent.compression_enabled = True
        agent._compression_feasibility_checked = True
        monkeypatch.setattr(agent.context_compressor, "should_compress", lambda *a, **k: agent.client.chat.completions.create.call_count == 2 and not compression_calls)
        def compact(messages, system, **kwargs):
            compression_calls.append(kwargs.get("trigger"))
            if control == "compression_interrupt":
                agent._interrupt_requested = True
            return [{"role": "user", "content": SUMMARY_PREFIX + "\nEarlier fixture history."}, {"role": "assistant", "content": "Historical context summarized."}] + [dict(m) for m in messages], system
        monkeypatch.setattr(agent, "_compress_context", compact)
        stages = [_tool(), _final(action), _final(action), _final(action)]
        expected_calls, expected_effects = (2 if control == "compression_interrupt" else 4), 1
    elif control == "safe_repeat":
        stages = [_tool(), _final(action), _tool(call_id="read_2"), _final(DONE)]
        expected_calls, expected_effects = 4, 2
    elif control == "uncertain_tool":
        agent.valid_tool_names = {"terminal"}
        agent.tools[0]["function"]["name"] = "terminal"
        tool_result = '{"error": "Cancelled after dispatch; outcome unknown"}'
        stages = [_tool("terminal"), _final(action), _final(action), _final(action)]
        expected_calls, expected_effects = 4, 1
    elif control == "large_history":
        # Production-shaped structured history: first/tail rows, long tool output,
        # Unicode paths, schema and a 4096-token output reserve within a 65536 window.
        history = [{"role": "user", "content": "Earlier authorized task."},
                   {"role": "assistant", "content": "", "tool_calls": [{"id": "prior_1", "type": "function", "function": {"name": "web_search", "arguments": "{}"}}]},
                   {"role": "tool", "tool_call_id": "prior_1", "content": "alpha beta gamma delta " * 9000},
                   {"role": "assistant", "content": "Earlier task complete."}]
        history += [{"role": role, "content": f"Protected tail {i}: /tmp/雪.txt"}
                    for i in range(10) for role in ("user", "assistant")]
    result, effects = _run(agent, user, stages, history, tool_result)
    assert agent.client.chat.completions.create.call_count == expected_calls
    assert effects == expected_effects
    if control in {"interrupt", "compression_interrupt"}:
        assert result["interrupted"] is True
        assert result["completed"] is False
    elif control.startswith("checkpoint_"):
        assert result["completed"] is False
        assert result["failed"] is True
        assert result.get("checkpoint_saved") is False
        assert "checkpoint" in result["final_response"].lower()
        assert any(word in result["final_response"].lower() for word in ("copy", "doctor", "storage"))
        assert result["messages"][-1]["content"] == result["final_response"]
        assert result["final_response"].count("Pending step:") == 1
    elif control in {"mutating", "counter_other_continue", "provider_error", "compression", "uncertain_tool"}:
        assert result["completed"] is False
        assert result["failed"] is True
        assert len([m for m in result["messages"] if m.get("role") == "tool"]) == 1
    else:
        assert result["completed"] is True
        expected = action if control in {"disabled", "no_tools"} else wait if control in {"steer", "redirect"} else DONE
        assert result["final_response"] == expected
    if control in {"compression", "compression_interrupt"}:
        assert compression_calls == ["pre_api"]
    if control == "large_history":
        assert agent.context_compressor.context_length == 65536
        sent = agent.client.chat.completions.create.call_args_list[-1].kwargs
        assert sum(len(str(m.get("content", ""))) for m in sent["messages"]) > 150000
        assert sent.get("max_tokens", sent.get("max_completion_tokens")) == 4096
    if control in {"steer", "redirect"}:
        from agent.context_compressor import ContextCompressor
        real_users = [m for m in result["messages"] if ContextCompressor._is_actionable_user_turn(m)
                      and not ContextCompressor._is_synthetic_compression_user_turn(m)]
        assert "Pause and wait" in real_users[-1]["content"]
