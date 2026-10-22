from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import model_tools
from hermes_cli.broker_mode import _QUALIFICATION_MODE_MARKER
from run_agent import AIAgent


def _tool_definition(name: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "hostile fixture",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _tool_response(call_id: str):
    call = SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name="terminal", arguments='{"command":"touch escaped"}'),
    )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="", tool_calls=[call]),
                finish_reason="tool_calls",
            )
        ],
        model="hostile/model",
        usage=None,
    )


def _text_response(content: str, finish_reason: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(content=content, tool_calls=None),
            finish_reason=finish_reason,
        )],
        model="hostile/model",
        usage=None,
    )


def test_qualification_mode_tool_definition_fast_path_skips_registry(monkeypatch):
    monkeypatch.setenv(_QUALIFICATION_MODE_MARKER, "1")
    monkeypatch.setattr(
        model_tools,
        "_compute_tool_definitions",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("registry reached")),
    )
    model_tools._last_resolved_tool_names = ["terminal"]

    assert model_tools.get_tool_definitions(quiet_mode=True) == []
    assert model_tools._last_resolved_tool_names == []


def test_qualification_mode_rejects_direct_tool_execution_before_dispatch(monkeypatch):
    monkeypatch.setenv(_QUALIFICATION_MODE_MARKER, "1")
    monkeypatch.setattr(
        model_tools,
        "coerce_tool_args",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("dispatch preparation reached")),
    )

    result = model_tools.handle_function_call("terminal", {"command": "touch escaped"})

    assert "disabled in qualification mode" in result


def test_real_agent_rejects_hostile_model_tool_calls_without_dispatch(monkeypatch):
    monkeypatch.setenv(_QUALIFICATION_MODE_MARKER, "1")
    with (
        patch("run_agent.get_tool_definitions", return_value=[_tool_definition("terminal")]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-qualification",
            base_url="http://127.0.0.1:11434/v1",
            provider="custom",
            model="hostile/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    assert agent.tools == []
    assert agent.valid_tool_names == set()
    agent._cached_system_prompt = "Return plain text."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = [
        _tool_response("hostile-1"),
        _tool_response("hostile-2"),
        _tool_response("hostile-3"),
    ]

    with (
        patch("run_agent.handle_function_call") as dispatch,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("Answer without tools")

    assert result["completed"] is False
    assert "invalid tool call" in result["error"].lower()
    dispatch.assert_not_called()
    for call in agent.client.chat.completions.create.call_args_list:
        assert not call.kwargs.get("tools")
        assert call.kwargs["temperature"] == 0
        assert call.kwargs["seed"] == 20260830
        assert call.kwargs["max_tokens"] == 256


def test_qualification_mode_overrides_conflicting_sampling_configuration(monkeypatch):
    monkeypatch.setenv(_QUALIFICATION_MODE_MARKER, "1")
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-qualification",
            base_url="http://127.0.0.1:11434/v1",
            provider="custom",
            model="hostile/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            max_tokens=8192,
            request_overrides={
                "temperature": 0.9,
                "seed": 7,
                "max_tokens": 9999,
                "unusual_supported_option": "kept",
            },
        )

    assert agent.max_tokens == 256
    assert agent.request_overrides == {
        "temperature": 0,
        "seed": 20260830,
        "unusual_supported_option": "kept",
    }


def test_qualification_length_continuation_never_raises_output_cap(monkeypatch):
    monkeypatch.setenv(_QUALIFICATION_MODE_MARKER, "1")
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-qualification",
            base_url="http://127.0.0.1:11434/v1",
            provider="custom",
            model="hostile/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent._cached_system_prompt = "Return plain text."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    agent.client = MagicMock()
    agent.client.chat.completions.create.side_effect = [
        _text_response("first", "length"),
        _text_response("second", "stop"),
    ]

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("Stay within the output cap")

    assert result["completed"] is True
    assert [
        call.kwargs["max_tokens"]
        for call in agent.client.chat.completions.create.call_args_list
    ] == [256, 256]
