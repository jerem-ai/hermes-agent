"""Actual turn/stream boundaries must enforce the passive contract without extra inference."""
from tests.agent.test_completion_announcement_recovery import _run, _final
from tests.agent import test_completion_announcement_recovery as helpers
from tests.agent.test_acceptance_contract import make_spec, attach
from agent.stream_delivery import StreamDeliveryMixin

guard_agent = helpers.guard_agent


def test_real_turn_rejects_docs_as_implementation_completion(guard_agent, tmp_path):
    agent, _ = guard_agent
    attach(agent, make_spec(tmp_path))
    result, effects = _run(agent, "Is the original implementation complete?",
                           [_final("Yes, the original task is complete. 687 tests passed.")])
    assert "parent objective remains open" in result["final_response"]
    assert "687 tests passed" not in result["final_response"]
    assert effects == 0
    assert agent.client.chat.completions.create.call_count == 1
    assert result["messages"][-1]["content"] == result["final_response"]


def test_unchecked_completion_does_not_stream(guard_agent, tmp_path):
    agent, _ = guard_agent
    attach(agent, make_spec(tmp_path))
    delivered = []
    agent.stream_delta_callback = delivered.append
    agent._stream_callback = None
    agent._stream_writer_superseded = lambda: False
    StreamDeliveryMixin._fire_stream_delta(agent, "Everything is done.")
    assert delivered == []


def test_ordinary_uncontracted_answer_still_finishes(guard_agent):
    agent, _ = guard_agent
    result, effects = _run(agent, "Report the existing checks.", [_final("All checks passed.")])
    assert result["final_response"] == "All checks passed."
    assert result["completed"] is True and effects == 0


def test_output_plugin_cannot_reintroduce_unverified_completion(guard_agent, tmp_path):
    from unittest.mock import patch
    agent, _ = guard_agent
    attach(agent, make_spec(tmp_path))
    with patch("agent.turn_finalizer.apply_llm_output_transform",
               return_value=("Everything is done.", True, {})):
        result, effects = _run(agent, "Report progress.", [_final("The document was edited.")])
    assert "parent objective remains open" in result["final_response"]
    assert effects == 0


def test_real_session_lineage_reloads_parent_contract(guard_agent, tmp_path):
    from types import SimpleNamespace
    from agent.acceptance_contract import load, report
    agent, db = guard_agent
    attach(agent, make_spec(tmp_path))
    db.ensure_session(agent.session_id, source="bluebubbles")
    db.create_session("acceptance-child", source="bluebubbles", parent_session_id=agent.session_id)
    child = SimpleNamespace(session_id="acceptance-child", _session_db=db)
    assert load(child) == load(agent)
    assert not report(child)["tasks"]["implementation"]["complete"]


def test_full_scale_native_compression_preserves_parent(guard_agent, tmp_path):
    from agent.acceptance_contract import context
    from tests.agent.test_compression_abort_state_reset import _InPlaceSuccessCompressor
    agent, db = guard_agent
    attach(agent, make_spec(tmp_path))
    db.ensure_session(agent.session_id, source="bluebubbles")
    messages = []
    for index in range(42):
        call = f"read-{index}"
        messages.extend([
            {"role": "user", "content": f"Inspect fixture {index}."},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": call, "type": "function",
                 "function": {"name": "web_search", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": call, "content": "diagnostic 雪 " * 450},
            {"role": "assistant", "content": "The fixture was read."},
        ])
    messages.extend([{"role": "user", "content": "Only correct the document now."},
                     {"role": "assistant", "content": "The documentation subtask was checked."}])
    assert len(messages) == 170 and sum(len(m["content"]) for m in messages) > 240_000
    agent.context_compressor = _InPlaceSuccessCompressor()
    agent._compression_feasibility_checked = True
    agent.compression_in_place = True
    compacted, _ = agent._compress_context(messages, "sys", approx_tokens=60_000, force=True)
    rendered = "\n".join(m.get("content", "") for m in compacted if isinstance(m.get("content"), str))
    assert "Actual caller behavior accepted" in rendered
    assert "Deliver working caller acceptance" in rendered
    assert "Actual caller behavior accepted" in context(agent)
    assert agent.client.chat.completions.create.call_count == 0


def test_yes_cannot_certify_the_original_assignment(guard_agent, tmp_path):
    agent, _ = guard_agent
    attach(agent, make_spec(tmp_path))
    result, effects = _run(agent, "So everything is done?", [_final("Yes.")])
    assert "parent objective remains open" in result["final_response"]
    assert effects == 0


def test_cli_status_resolves_compressed_session_lineage(guard_agent, tmp_path, monkeypatch, capsys):
    import json
    import sys
    from hermes_cli.acceptance_contract import main
    agent, db = guard_agent
    attach(agent, make_spec(tmp_path))
    db.ensure_session(agent.session_id, source="bluebubbles")
    db.create_session("cli-child", source="bluebubbles", parent_session_id=agent.session_id)
    monkeypatch.setenv("HERMES_HOME", str(db.db_path.parent))
    monkeypatch.setattr(sys, "argv", ["acceptance-contract", "--session", "cli-child", "status"])
    main()
    output = capsys.readouterr().out
    report = json.loads(output[output.index("{"):])
    assert report["tasks"]["implementation"]["complete"] is False
