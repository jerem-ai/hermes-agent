"""Replay the saved ROM-hub stop through the actual turn loop without inference."""
import pytest

from tests.agent import test_completion_announcement_recovery as helpers
from tests.agent.test_completion_announcement_recovery import _run, _final, _tool, DONE

guard_agent = helpers.guard_agent
USER = "Set up the ROM library from the NAS for the authorized network clients."
STOPPED = (
    "Nice library \u2014 618GB across 55+ systems (NES, SNES, N64, Game Boy, "
    "Genesis, PS2, Switch, and a ton more). Bazzite's got 62 GiB RAM and Flatpak ready. "
    "So it has everything needed to host this properly.\n\n"
    "I'm mapping out the ROM hub setup on Bazzite with NAS integration \u2014 "
    "give me a minute to check what's available..."
)


def _history():
    return [
        {"role": "user", "content": "Earlier authorized work in /tmp/雪."},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "old", "type": "function", "function": {"name": "web_search", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "old", "content": "alpha beta gamma delta " * 9000},
        {"role": "assistant", "content": "Earlier work is complete."},
    ] + [{"role": role, "content": f"Protected tail {i}: /tmp/雪.txt"}
         for i in range(10) for role in ("user", "assistant")]


@pytest.mark.parametrize("boundary", ["post_tool", "large_context", "compression", "streaming", "interrupt"])
def test_progress_promise_continues_the_actual_tool_loop(guard_agent, monkeypatch, boundary):
    agent, db = guard_agent
    history = _history() if boundary in {"large_context", "compression"} else None
    compressed = []
    if boundary == "compression":
        from agent.context_compressor import SUMMARY_PREFIX
        agent.compression_enabled = True
        agent._compression_feasibility_checked = True
        monkeypatch.setattr(agent.context_compressor, "should_compress",
                            lambda *a, **k: agent.client.chat.completions.create.call_count == 2 and not compressed)
        def compact(messages, system, **kwargs):
            compressed.append(kwargs.get("trigger"))
            return [{"role": "user", "content": SUMMARY_PREFIX + "\nOlder fixture work."},
                    {"role": "assistant", "content": "Historical context summarized."}] + messages[len(history):], system
        monkeypatch.setattr(agent, "_compress_context", compact)
    if boundary == "interrupt":
        import agent.turn_final_response as phase
        original = phase._queued_user_control
        def cancel(owner):
            owner._interrupt_requested = True
            return original(owner)
        monkeypatch.setattr(phase, "_queued_user_control", cancel)
    stages = [_tool(), _final(STOPPED), _tool(call_id="continued"), _final(DONE)]
    if boundary == "streaming":
        from tests.agent.test_streaming import _make_stream_chunk, _make_tool_call_delta
        delivered = []
        agent.stream_delta_callback = delivered.append
        def tool_stream(call_id):
            return iter([_make_stream_chunk(tool_calls=[_make_tool_call_delta(
                tc_id=call_id, name="web_search", arguments="{}")]),
                _make_stream_chunk(finish_reason="tool_calls")])
        def text_stream(text):
            return iter([_make_stream_chunk(content=text[:20]),
                         _make_stream_chunk(content=text[20:], finish_reason="stop")])
        stages = [tool_stream("first"), text_stream(STOPPED), tool_stream("continued"), text_stream(DONE)]
    result, effects = _run(agent, USER, stages, history)
    if boundary == "interrupt":
        assert result["interrupted"] is True and result["completed"] is False
        assert agent.client.chat.completions.create.call_count == 2
        assert effects == 1
        return
    assert result["final_response"] == DONE and result["completed"] is True
    assert effects == 2 and agent.client.chat.completions.create.call_count == 4
    requests = [c.kwargs for c in agent.client.chat.completions.create.call_args_list]
    assert all(r["tools"] == requests[0]["tools"] for r in requests)
    systems = [[m for m in r["messages"] if m["role"] == "system"] for r in requests]
    if boundary == "compression":
        assert systems[0] == systems[1] and systems[2] == systems[3]
    else:
        assert all(s == systems[0] for s in systems)
    assert db.get_messages_as_conversation(agent.session_id)[-1]["content"] == DONE
    if history:
        assert sum(len(str(m.get("content", ""))) for m in requests[0]["messages"]) > 198000
        assert agent.context_compressor.context_length == 65536
    if boundary == "compression":
        assert compressed == ["pre_api"]
    if boundary == "streaming":
        assert delivered and all(r["stream"] for r in requests)


@pytest.mark.parametrize("text", [
    STOPPED,
    "Verified details. " * 80 + "Give me a moment to inspect /tmp/雪.txt...",
    "Hang tight while I check the available packages.",
    "Bear with me while I verify the mounted library.",
])
def test_pending_work_is_bounded_and_stays_failed_after_output_plugin(guard_agent, monkeypatch, text):
    agent, db = guard_agent
    def hooks(name, *args, **kwargs):
        return ["Everything is complete."] if name == "transform_llm_output" else []
    monkeypatch.setattr("agent.turn_finalizer._invoke_hook_safely", hooks)
    result, effects = _run(agent, USER, [_tool(), _final(text), _final(text), _final(text)])
    assert effects == 1 and agent.client.chat.completions.create.call_count == 4
    assert result["completed"] is False and result["failed"] is True
    assert result["checkpoint_saved"] is True
    assert "Pending step:" in result["final_response"]
    assert db.get_messages_as_conversation(agent.session_id)[-1]["content"] == result["final_response"]


@pytest.mark.parametrize("user,text", [
    (USER, 'The old log ended with "give me a minute to check what is available..."'),
    (USER, "The old log ended with `give me a minute to inspect /tmp/雪...`"),
    (USER, "The subtask is complete. The report quotes: ‘Give me a moment to inspect /tmp/雪...’"),
    (USER, "After you approve, give me a minute to check the separate account."),
    ("Translate to English: Dame un minuto para verificar.", "Give me a minute to check."),
    ("Give an example sentence.", "Hang tight while I check the available packages."),
    ("Pause the task.", "Give me a minute to check what is available..."),
    (USER, "The requested checks passed. No further work is pending."),
    (USER, "Give me a minute to explain: the verified answer is 42."),
])
def test_unusual_correct_progress_wording_stays_final(guard_agent, user, text):
    agent, _ = guard_agent
    result, effects = _run(agent, user, [_tool(), _final(text)])
    assert result["completed"] is True and result["final_response"] == text
    assert effects == 1 and agent.client.chat.completions.create.call_count == 2


def test_legitimate_output_plugin_still_transforms_and_persists(guard_agent, monkeypatch):
    agent, db = guard_agent
    text = 'The subtask is complete. The old log said "give me a minute to check..."'
    def hooks(name, *args, **kwargs):
        return [text] if name == "transform_llm_output" else []
    monkeypatch.setattr("agent.turn_finalizer._invoke_hook_safely", hooks)
    result, _ = _run(agent, "Report the completed subtask.", [_final(DONE)])
    assert result["completed"] is True and result["response_transformed"] is True
    assert result["final_response"] == text
    assert db.get_messages_as_conversation(agent.session_id)[-1]["content"] == text
