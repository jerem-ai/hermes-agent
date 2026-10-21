"""Repeat compaction after the saved 39-row, 58K-request boundary failure."""
from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import ContextCompressor, SUMMARY_PREFIX, _SUMMARY_END_MARKER
from tests.agent import test_completion_announcement_recovery as helpers
from tests.agent.test_completion_announcement_recovery import _tool, _final, DONE

guard_agent = helpers.guard_agent

TASK = "Keep the original ROM library objective: Apple TV, Bluetooth controller, and the complete catalog."
SIZES = [(187, 86), (119, 1881), (0, 1941), (0, 716), (307, 1092), (244, 1820),
         (0, 1627), (115, 1927), (699, 230), (146, 2126), (72, 562), (81, 1151),
         (121, 601), (172, 502), (37, 82), (0, 2056), (128, 2182), (113, 16479)]


def fixture_messages():
    summary = SUMMARY_PREFIX + "\n" + "Prior durable facts. " * 760 + "\n" + _SUMMARY_END_MARKER
    messages = [{"role": "user", "content": summary},
                {"role": "assistant", "content": "Previous verified subtask. " * 45},
                {"role": "user", "content": TASK}]
    for i, (assistant_size, tool_size) in enumerate(SIZES):
        call = f"read-{i}"
        messages.extend([
            {"role": "assistant", "content": "a" * assistant_size,
             "tool_calls": [{"id": call, "type": "function",
                             "function": {"name": "web_search", "arguments": '{"query":"' + "x" * 350 + '"}'}}]},
            {"role": "tool", "tool_call_id": call, "content": "e" * tool_size},
        ])
    assert len(messages) == 39
    return messages


def compressor():
    return ContextCompressor(model="leyline/tars", config_context_length=65536,
                             threshold_percent=0.8, summary_target_ratio=0.2,
                             protect_first_n=3, protect_last_n=20, tail_mode="lean", quiet_mode=True)


def response():
    r = MagicMock()
    r.choices[0].message.content = "Durable earlier work. The original task remains open."
    return r


@pytest.mark.parametrize("resumed", [False, True])
def test_pressure_recompacts_real_post_handoff_work(resumed):
    cc = compressor()
    cc.compression_count = 0 if resumed else 1
    messages = fixture_messages()
    before = deepcopy(messages)
    with patch("agent.context_compressor.call_llm", return_value=response()) as summary:
        result = cc.compress(messages, current_tokens=58493)
    assert summary.call_count == 1
    assert result != before
    assert cc._last_compression_made_progress is True
    assert cc._structural_no_op_backoff_until == 0
    assert cc.context_length == 65536 and cc.threshold_percent == 0.8 and cc.summary_target_ratio == 0.2
    assert any(TASK in str(m.get("content", "")) for m in result)
    assert any(m.get("tool_call_id") == "read-17" for m in result)
    calls = {c["id"] for m in result for c in m.get("tool_calls", [])}
    assert all(m.get("tool_call_id") in calls for m in result if m.get("role") == "tool")
    assert messages == before


@pytest.mark.parametrize("case", ["below_trigger", "multimodal_request", "multiple_user_anchor", "indivisible_tool_group"])
def test_pressure_does_not_override_required_or_affordable_tail(case):
    cc = compressor()
    cc.compression_count = 1
    messages = fixture_messages()
    pressure = 58493
    if case == "below_trigger":
        pressure = 50000
    elif case == "multimodal_request":
        messages[2]["content"] = [{"type": "text", "text": TASK},
                                   {"type": "image_url", "image_url": {"url": "https://example.invalid/fixture.png"}}]
    elif case == "multiple_user_anchor":
        cc.min_tail_user_messages = 2
        messages.insert(1, {"role": "assistant", "content": "Prior task."})
        messages.insert(2, {"role": "user", "content": "Keep this second user anchor verbatim too."})
    else:
        # The only action is a still-unread atomic group. Splitting it would invent savings.
        messages = messages[:3] + messages[-2:]
    before = deepcopy(messages)
    with patch("agent.context_compressor.call_llm", return_value=response()) as summary:
        result = cc.compress(messages, current_tokens=pressure)
    if case == "multiple_user_anchor":
        assert any("Keep this second user anchor verbatim too." in str(m.get("content", "")) for m in result)
        assert any(TASK in str(m.get("content", "")) for m in result)
        return
    summary.assert_not_called()
    assert result == before
    assert cc._ineffective_compression_count == 0


@pytest.mark.parametrize("streaming", [False, True])
def test_native_turn_recompresses_and_persists_with_fixed_prompt_overhead(guard_agent, streaming):
    from agent.iteration_budget import IterationBudget
    agent, db = guard_agent
    agent.max_iterations = 40
    agent.iteration_budget = IterationBudget(40)
    agent.context_compressor = compressor()
    agent.context_compressor.compression_count = 1
    agent.compression_enabled = True
    agent.compression_in_place = True
    agent._compression_feasibility_checked = True
    agent._cached_system_prompt = "Keep the original user objective and resource limits. " + "fixed policy text " * 10000
    fixture = fixture_messages()
    stages = [_tool(call_id=f"live-{i}") for i in range(len(SIZES))] + [_final(DONE)]
    outputs = ["e" * size for _, size in SIZES]
    if streaming:
        from tests.agent.test_streaming import _make_stream_chunk, _make_tool_call_delta
        delivered = []
        agent.stream_delta_callback = delivered.append
        stages = [iter([_make_stream_chunk(tool_calls=[_make_tool_call_delta(
            tc_id=f"live-{i}", name="web_search", arguments="{}")]),
            _make_stream_chunk(finish_reason="tool_calls")]) for i in range(len(SIZES))]
        stages.append(iter([_make_stream_chunk(content=DONE, finish_reason="stop")]))
    agent.client.chat.completions.create.side_effect = stages
    with (patch("model_tools.get_tool_definitions", return_value=deepcopy(agent.tools)),
          patch("model_tools.handle_function_call", side_effect=outputs),
          patch("agent.context_compressor.call_llm", return_value=response()) as summary,
          patch.object(agent, "_save_trajectory"), patch.object(agent, "_cleanup_task_resources")):
        result = agent.run_conversation(TASK, conversation_history=fixture[:2])
    assert result["completed"] is True and result["final_response"] == DONE
    assert summary.call_count >= 1 and agent.context_compressor.compression_count >= 2
    assert agent.session_id == db.get_session(agent.session_id)["id"]
    saved = db.get_messages_as_conversation(agent.session_id)
    assert saved[-1]["content"] == DONE
    assert any(TASK in str(m.get("content", "")) for m in saved)
    assert any(m.get("tool_call_id") == "live-17" for m in saved)
    requests = [c.kwargs for c in agent.client.chat.completions.create.call_args_list]
    assert len(requests) == 19 and all(r["tools"] == requests[0]["tools"] for r in requests)
    assert sum(len(str(m.get("content", ""))) for m in requests[0]["messages"]) > 190000
