"""Replay the Desktop key-setup stop with mocked providers and native persistence."""
import pytest

from tests.agent import test_completion_announcement_recovery as helpers
from tests.agent.test_completion_announcement_recovery import _run, _final, _tool, DONE

guard_agent = helpers.guard_agent
USER = "I wanted YOU to be able to access bazzite from scout, where you live, since it's basically your computer"
STOPPED = (
    "Good. The host config exists and uses the existing user key.\n\n"
    "Let me add the hermie key so Hermie can authenticate to Bazzite:"
)


@pytest.mark.parametrize("boundary", ["post_tool", "large_context", "compression", "interrupt"])
def test_file_action_promise_continues_after_tool_work(guard_agent, monkeypatch, boundary):
    agent, db = guard_agent
    history = None
    compressed = []
    if boundary in {"large_context", "compression"}:
        history = [{"role": "user", "content": "Earlier authorized work in /tmp/雪."},
                   {"role": "assistant", "content": "", "tool_calls": [{"id": "old", "type": "function",
                    "function": {"name": "web_search", "arguments": "{}"}}]},
                   {"role": "tool", "tool_call_id": "old", "content": "alpha beta gamma delta " * 9000},
                   {"role": "assistant", "content": "Earlier work is complete."}]
    if boundary == "compression":
        from agent.context_compressor import SUMMARY_PREFIX
        agent.compression_enabled = True
        agent._compression_feasibility_checked = True
        monkeypatch.setattr(agent.context_compressor, "should_compress",
                            lambda *a, **k: agent.client.chat.completions.create.call_count == 2 and not compressed)
        def compact(messages, system, **kwargs):
            compressed.append(kwargs.get("trigger"))
            return [{"role": "user", "content": SUMMARY_PREFIX + "\nOlder fixture context."},
                    {"role": "assistant", "content": "Historical context summarized."}] + messages[len(history):], system
        monkeypatch.setattr(agent, "_compress_context", compact)
    if boundary == "interrupt":
        import agent.turn_final_response as phase
        original = phase._queued_user_control
        def cancel(owner):
            owner._interrupt_requested = True
            return original(owner)
        monkeypatch.setattr(phase, "_queued_user_control", cancel)
    result, effects = _run(agent, USER, [_tool(), _final(STOPPED), _tool(call_id="continued"), _final(DONE)], history)
    if boundary == "interrupt":
        assert result["interrupted"] is True
        assert result["completed"] is False
        assert effects == 1
        return
    assert result["completed"] is True
    assert result["final_response"] == DONE
    assert effects == 2
    assert agent.client.chat.completions.create.call_count == 4
    requests = [call.kwargs for call in agent.client.chat.completions.create.call_args_list]
    assert all(r["tools"] == requests[0]["tools"] for r in requests)
    systems = [[m for m in r["messages"] if m["role"] == "system"] for r in requests]
    if boundary == "compression":
        assert systems[0] == systems[1]
        assert systems[2] == systems[3]
    else:
        assert all(s == systems[0] for s in systems)
    assert db.get_messages_as_conversation(agent.session_id)[-1]["content"] == DONE
    if boundary == "large_context":
        assert sum(len(str(m.get("content", ""))) for m in requests[0]["messages"]) > 190000
    if boundary == "compression":
        assert compressed == ["pre_api"]


def test_unfulfilled_key_promise_exhausts_as_incomplete(guard_agent):
    agent, db = guard_agent
    result, effects = _run(agent, USER, [_tool(), _final(STOPPED), _final(STOPPED), _final(STOPPED)])
    assert effects == 1
    assert result["completed"] is False
    assert result["failed"] is True
    assert result["checkpoint_saved"] is True
    assert agent.client.chat.completions.create.call_count == 4
    assert "Pending step:" in db.get_messages_as_conversation(agent.session_id)[-1]["content"]


def test_streamed_file_action_promise_keeps_the_tool_loop_running(guard_agent):
    from tests.agent.test_streaming import _make_stream_chunk, _make_tool_call_delta
    agent, db = guard_agent
    chunks = []
    agent.stream_delta_callback = chunks.append
    def tool_stream(call_id):
        return iter([_make_stream_chunk(tool_calls=[_make_tool_call_delta(
            tc_id=call_id, name="web_search", arguments="{}")]),
            _make_stream_chunk(finish_reason="tool_calls")])
    def text_stream(text):
        return iter([_make_stream_chunk(content=text[:20]),
                     _make_stream_chunk(content=text[20:], finish_reason="stop")])
    result, effects = _run(agent, USER, [tool_stream("first"), text_stream(STOPPED),
                                       tool_stream("continued"), text_stream(DONE)])
    assert result["completed"] is True
    assert result["final_response"] == DONE
    assert effects == 2
    assert chunks
    assert all(c.kwargs.get("stream") for c in agent.client.chat.completions.create.call_args_list)
    assert agent.client.chat.completions.create.call_count == 4
    assert db.get_messages_as_conversation(agent.session_id)[-1]["content"] == DONE


@pytest.mark.parametrize("text", [
    "Let me append the verified line to /tmp/雪/config:",
    "Let me configure the selected identity:",
    "Let me create the missing config file:",
    "Let me update the host entry:",
])
def test_sibling_file_actions_are_pending(guard_agent, text):
    agent, _ = guard_agent
    result, _ = _run(agent, "Finish the authorized file edit.", [_final(text), _tool(), _final(DONE)])
    assert result["final_response"] == DONE
    assert agent.client.chat.completions.create.call_count == 3


@pytest.mark.parametrize("user,text", [
    (USER, "Existing SSH access works. No further setup is needed."),
    (USER, "Let me add one clarification: access already works."),
    (USER, "Let me add that the existing identity is valid."),
    (USER, "After you approve, let me add the separate identity:"),
    ("Translate to English: Déjame añadir la clave.", "Let me add the key."),
    ("Give an example sentence.", "Let me configure the selected identity:"),
])
def test_valid_unusual_answers_stay_final(guard_agent, user, text):
    agent, _ = guard_agent
    result, effects = _run(agent, user, [_tool(), _final(text)])
    assert result["final_response"] == text
    assert agent.client.chat.completions.create.call_count == 2
    assert effects == 1
