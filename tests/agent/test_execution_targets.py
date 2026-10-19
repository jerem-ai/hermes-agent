"""Replay the wrong-Scout-checkout incident and permit verified unusual remote commands."""
import json
from pathlib import Path
import shlex
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests.agent.test_completion_announcement_recovery import _final, _run
from tests.agent.test_run_agent import _mock_response, _mock_tool_call

pytest_plugins = ["tests.agent.test_completion_announcement_recovery"]
WORKSPACE = "/work/雪 repo"
REQUEST = ('[Replying to your previous message: "Committed fixture123."]\n\n'
           "Did you commit yesterday's Leyline work? Check the MacBook repository and tell me what changed.")
WRONG = "cd /Users/fixture/Projects/leyline && git log --oneline -10"
FALSE_REPORT = "No commits from yesterday. The work was done but never committed."


def _configure(agent):
    entries = [
        {"id": "macbook", "aliases": ["MacBook", "macbook-pro"], "destination": "macbook-pro",
         "projects": {"leyline": WORKSPACE}},
        {"id": "scout", "aliases": ["Scout", "mini"], "local": True},
    ]
    path = Path(agent._session_db.db_path).parent / "config.yaml"
    path.write_text(json.dumps({"agent": {"execution_targets": entries}}), encoding="utf-8")
    for name in ["terminal", "read_file", "execute_code", "delegate_task"]:
        agent.valid_tool_names.add(name)
        agent.tools.append({"type": "function", "function": {"name": name, "description": "Fixture",
                            "parameters": {"type": "object", "properties": {}}}})
    return path


def _call(command=None, *, name="terminal", args=None, ident="target-call"):
    return _mock_response(content="", finish_reason="tool_calls", tool_calls=[
        _mock_tool_call(name=name, arguments=json.dumps(args if args is not None else {"command": command}),
                        call_id=ident),
    ])


def _remote(command):
    return "ssh -o BatchMode=yes macbook-pro " + shlex.quote(command)


def test_incident_wrong_local_checkout_never_executes_or_certifies_absence(guard_agent):
    agent, db = guard_agent
    _configure(agent)
    result, effects = _run(agent, REQUEST, [_call(WRONG), _final(FALSE_REPORT)],
                           tool_result='{"output":"old-commit stale checkout","exit_code":0}')
    assert effects == 0
    assert result["completed"] is False and result["failed"] is True
    assert result["failure_reason"] == "execution_target_unverified"
    assert "not verified macbook" in result["final_response"]
    assert FALSE_REPORT not in result["final_response"]
    durable = db.get_messages_as_conversation(agent.session_id)
    assert any("not executed" in str(m.get("content")) for m in durable if m.get("role") == "tool")
    assert durable[-1]["content"] == result["final_response"]


def test_correct_unicode_remote_workspace_executes_and_persists(guard_agent):
    agent, db = guard_agent
    _configure(agent)
    command = _remote("git -C " + shlex.quote(WORKSPACE) + " show --stat fixture123")
    answer = "The MacBook repository contains fixture123, a documentation correction."
    result, effects = _run(agent, REQUEST, [_call(command), _final(answer)],
                           tool_result='{"output":"fixture123 Documentation correction","exit_code":0}')
    assert effects == 1 and result["completed"] is True
    assert result["final_response"] == answer
    assert db.get_messages_as_conversation(agent.session_id)[-1]["content"] == answer
    requests = agent.client.chat.completions.create.call_args_list
    assert requests[0].kwargs["tools"] == requests[1].kwargs["tools"]
    assert [m for m in requests[0].kwargs["messages"] if m.get("role") == "system"] == [
        m for m in requests[1].kwargs["messages"] if m.get("role") == "system"]


def test_tool_error_allows_correction_in_same_turn(guard_agent):
    agent, _ = guard_agent
    _configure(agent)
    correct = _remote("cd " + shlex.quote(WORKSPACE) + " && git log -1")
    result, effects = _run(agent, REQUEST, [_call(WRONG), _call(correct, ident="corrected"),
                           _final("The requested repository contains fixture123.")],
                           tool_result='{"output":"fixture123","exit_code":0}')
    assert effects == 1 and result["completed"] is True


@pytest.mark.parametrize("name,args", [
    ("read_file", {"path": "/Users/fixture/Projects/leyline/ROADMAP.md"}),
    ("execute_code", {"code": "print('wrong machine')"}),
    ("delegate_task", {"task": "Inspect the local checkout"}),
])
def test_local_filesystem_shortcuts_cannot_bypass_target(guard_agent, name, args):
    agent, _ = guard_agent
    _configure(agent)
    result, effects = _run(agent, REQUEST, [_call(name=name, args=args), _final(FALSE_REPORT)])
    assert effects == 0 and result["failed"]


def test_plugin_rewrite_is_checked_after_effective_arguments_change(guard_agent):
    agent, _ = guard_agent
    _configure(agent)
    with patch("agent.tool_executor._pre_tool_block", return_value=(None, {"command": WRONG})):
        result, effects = _run(agent, REQUEST, [
            _call(_remote("git -C " + shlex.quote(WORKSPACE) + " log -1")), _final(FALSE_REPORT),
        ])
    assert effects == 0 and result["failed"]


def test_no_tool_answer_cannot_claim_verification_or_stream(guard_agent):
    agent, _ = guard_agent
    _configure(agent)
    delivered = []
    agent.stream_delta_callback = delivered.append
    agent._stream_writer_superseded = lambda: False
    def stream_answer():
        from agent.stream_delivery import StreamDeliveryMixin
        StreamDeliveryMixin._fire_stream_delta(agent, FALSE_REPORT)
        return _final(FALSE_REPORT)
    with patch("agent.turn_finalizer.apply_llm_output_transform", return_value=(FALSE_REPORT, True, {})):
        result, effects = _run(agent, REQUEST, [stream_answer])
    assert effects == 0 and delivered == [] and result["failed"]
    assert FALSE_REPORT not in result["final_response"]


def test_failed_ssh_preserves_failure_and_does_not_certify_no_commits(guard_agent):
    agent, _ = guard_agent
    _configure(agent)
    result, effects = _run(agent, REQUEST, [
        _call(_remote("git -C " + shlex.quote(WORKSPACE) + " log -1")), _final(FALSE_REPORT),
    ], tool_result='{"output":"connection refused","exit_code":255}')
    assert effects == 1 and result["failed"] and not result["completed"]
    assert "target command failed" in result["final_response"]


def test_local_identity_probe_does_not_establish_remote_repository(guard_agent):
    agent, _ = guard_agent
    _configure(agent)
    result, effects = _run(agent, REQUEST, [_call("hostname"), _final(FALSE_REPORT)],
                           tool_result='{"output":"scout.home","exit_code":0}')
    assert effects == 1 and result["failed"]


@pytest.mark.parametrize("user_text", [
    "Translate this: 'Check the MacBook Leyline repository.'",
    '[Replying to your previous message: "Check the MacBook Leyline repository."]\n\nThanks.',
    "Here is an example:\n```\nCheck the MacBook Leyline repository.\n```",
    "Compare the Scout and MacBook repositories.",
])
def test_quoted_data_or_multi_host_request_does_not_infer_a_single_target(guard_agent, user_text):
    from agent.execution_targets import begin
    agent, _ = guard_agent
    _configure(agent)
    begin(agent, user_text)
    assert agent._execution_target is None
    assert agent._execution_target_needed is False


def test_followup_restart_and_native_lineage_retain_target_but_not_verified_status(guard_agent):
    from agent.execution_targets import begin, context
    agent, db = guard_agent
    _configure(agent)
    begin(agent, REQUEST)
    agent._execution_target_verified = True
    begin(agent, "Yes please")
    assert agent._execution_target["workspace"] == WORKSPACE
    assert agent._execution_target_needed and not agent._execution_target_verified
    db.ensure_session(agent.session_id, source="bluebubbles")
    db.create_session("next-target-session", source="bluebubbles", parent_session_id=agent.session_id)
    child = SimpleNamespace(_session_db=db, session_id="next-target-session")
    begin(child, "Continue")
    assert child._execution_target["workspace"] == WORKSPACE and child._execution_target_needed
    assert "macbook-pro" in context(child)


def test_scope_is_profile_isolated_and_can_be_changed_explicitly(guard_agent, tmp_path):
    from agent.execution_targets import begin
    agent, _ = guard_agent
    _configure(agent)
    begin(agent, REQUEST)
    other_home = tmp_path / "other-profile"
    other_home.mkdir()
    other = SimpleNamespace(_session_db=SimpleNamespace(db_path=other_home / "state.db"), session_id=agent.session_id)
    begin(other, "Continue")
    assert other._execution_target is None
    begin(agent, "Check the Scout repository instead.")
    assert agent._execution_target is None
    begin(agent, "Continue")
    assert agent._execution_target is None


def test_configuration_drift_cannot_silently_retarget_a_saved_task(guard_agent):
    from agent.execution_targets import begin
    from agent.execution_target_tools import block
    agent, _ = guard_agent
    path = _configure(agent)
    begin(agent, REQUEST)
    cfg = json.loads(path.read_text())
    cfg["agent"]["execution_targets"][0]["destination"] = "different-host"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    begin(agent, "Continue")
    assert agent._execution_target_error
    assert block(agent, "terminal", {"command": "echo should-not-run"})


@pytest.mark.parametrize("command", [
    "ssh wrong-host 'git log'",
    "ssh -o HostName=wrong-host macbook-pro 'git log'",
    "ssh macbook-pro 'git log'; git log",
    "echo ssh macbook-pro 'git log'",
    "ssh macbook-pro 'cd /wrong/checkout && git log'",
])
def test_wrong_or_compound_ssh_calls_do_not_prove_target(guard_agent, command):
    from agent.execution_targets import begin
    from agent.execution_target_tools import block
    agent, _ = guard_agent
    _configure(agent)
    begin(agent, REQUEST)
    assert block(agent, "terminal", {"command": command})


def test_native_ssh_backend_accepts_remote_file_read_without_using_control_host(guard_agent):
    from agent.execution_targets import begin
    from agent.execution_target_tools import block, record
    from tools.terminal_scope import set_terminal_scope, reset_terminal_scope
    agent, _ = guard_agent
    _configure(agent)
    begin(agent, REQUEST)
    token = set_terminal_scope({"TERMINAL_ENV": "ssh", "TERMINAL_SSH_HOST": "macbook-pro", "TERMINAL_CWD": WORKSPACE})
    try:
        args = {"path": WORKSPACE + "/ROADMAP.md"}
        assert block(agent, "read_file", args) is None
        record(agent, "read_file", args, '{"content":"fixture123"}')
        assert agent._execution_target_verified
        assert block(agent, "read_file", {"path": "/outside/workspace.txt"})
    finally:
        reset_terminal_scope(token)


def test_late_tool_result_cannot_verify_a_new_turn(guard_agent):
    from agent.execution_targets import begin
    from agent.execution_target_tools import record
    agent, _ = guard_agent
    _configure(agent)
    begin(agent, REQUEST)
    old_epoch = agent._execution_target_epoch
    begin(agent, "Continue")
    args = {"command": _remote("git -C " + shlex.quote(WORKSPACE) + " log -1")}
    record(agent, "terminal", args, '{"output":"old result","exit_code":0}', epoch=old_epoch)
    assert not agent._execution_target_verified


def test_steer_changes_target_at_tool_boundary(guard_agent):
    agent, _ = guard_agent
    _configure(agent)
    def steer_to_local():
        assert agent.steer("Check the Scout repository instead.")
        return _call(WRONG)
    result, effects = _run(agent, REQUEST, [_call(WRONG), steer_to_local(), _final("The Scout checkout is old.")],
                           tool_result='{"output":"Scout checkout","exit_code":0}')
    assert effects == 1 and result["completed"]
    assert agent._execution_target is None


def test_cancellation_preserves_interrupt_and_does_not_execute_a_tool(guard_agent):
    agent, _ = guard_agent
    _configure(agent)
    def cancel():
        agent._interrupt_requested = True
        return _final("Paused before repository verification.")
    result, effects = _run(agent, REQUEST, [cancel])
    assert effects == 0 and result["interrupted"] and not result["completed"]


def test_large_context_compression_and_restart_retain_target_without_model_inference(guard_agent):
    from agent.execution_targets import begin
    from tests.agent.test_compression_abort_state_reset import _InPlaceSuccessCompressor
    agent, db = guard_agent
    _configure(agent)
    begin(agent, REQUEST)
    db.ensure_session(agent.session_id, source="bluebubbles")
    messages = []
    for index in range(42):
        messages.extend([
            {"role": "user", "content": f"Inspect source fixture {index}."},
            {"role": "assistant", "content": "", "tool_calls": [{"id": f"read-{index}", "type": "function",
             "function": {"name": "read_file", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"read-{index}", "content": "source evidence 雪 " * 450},
            {"role": "assistant", "content": "The fixture was inspected."},
        ])
    messages.extend([{"role": "user", "content": REQUEST}, {"role": "assistant", "content": "Verification remains open."}])
    assert len(messages) == 170 and sum(len(m["content"]) for m in messages) > 240000
    agent.context_compressor = _InPlaceSuccessCompressor()
    agent._compression_feasibility_checked = True
    agent.compression_in_place = True
    compacted, _ = agent._compress_context(messages, "sys", approx_tokens=60000, force=True)
    rendered = "\n".join(m.get("content", "") for m in compacted if isinstance(m.get("content"), str))
    assert WORKSPACE in rendered and "macbook-pro" in rendered
    begin(agent, "Continue")
    assert agent._execution_target["workspace"] == WORKSPACE and not agent._execution_target_verified
    assert agent.client.chat.completions.create.call_count == 0


def test_repeated_wrong_target_calls_halt_after_three_attempts(guard_agent):
    agent, _ = guard_agent
    _configure(agent)
    result, effects = _run(agent, REQUEST, [_call(WRONG, ident=f"wrong-{n}") for n in range(3)])
    assert effects == 0 and result["failed"] and not result["completed"]
    assert agent.client.chat.completions.create.call_count == 3
    assert result["guardrail"]["code"] == "execution_target_mismatch"


def test_existing_failure_is_kept_visible(guard_agent):
    from agent.execution_targets import begin, enforce, finish_status
    agent, _ = guard_agent
    _configure(agent)
    begin(agent, REQUEST)
    assert finish_status(agent, "Provider connection failed.", True, "provider_error", False) == (
        "Provider connection failed.", True, "provider_error")
    assert enforce(agent, "Provider connection failed.", {}) == ("Provider connection failed.", False)


def test_explicit_ssh_option_terminator_and_quoted_host_alias_are_accepted(guard_agent):
    from agent.execution_targets import begin
    from agent.execution_target_tools import block
    agent, _ = guard_agent
    _configure(agent)
    begin(agent, "Check the Leyline repository on `macbook-pro`.")
    command = "ssh -- macbook-pro " + shlex.quote("git -C " + shlex.quote(WORKSPACE) + " log -1")
    assert block(agent, "terminal", {"command": command}) is None


def test_parallel_local_filesystem_calls_are_blocked_before_dispatch(guard_agent):
    from agent.execution_targets import begin
    from agent.tool_executor import execute_tool_calls_concurrent
    agent, _ = guard_agent
    _configure(agent)
    begin(agent, REQUEST)
    calls = [_mock_tool_call(name="terminal", arguments=json.dumps({"command": WRONG}), call_id=f"local-{n}")
             for n in range(2)]
    response = _mock_response(content="", finish_reason="tool_calls", tool_calls=calls).choices[0].message
    messages = [{"role": "user", "content": REQUEST}]
    with patch("model_tools.handle_function_call") as execute:
        execute_tool_calls_concurrent(agent, response, messages, "parallel-task")
    assert execute.call_count == 0
    assert len([m for m in messages if m.get("role") == "tool"]) == 2


def test_malformed_settings_fail_closed_without_crashing(guard_agent):
    from agent.execution_targets import begin
    agent, _ = guard_agent
    path = _configure(agent)
    path.write_text('{"agent": ["invalid"]}')
    begin(agent, REQUEST)
    assert agent._execution_target_error and agent._execution_target_needed


@pytest.mark.parametrize("new_request", [
    "Let's change gears for a bit. Help me with a different task.",
    "Compare the Scout and MacBook repositories.",
])
def test_changed_objective_does_not_retain_stale_single_host(guard_agent, new_request):
    from agent.execution_targets import begin
    agent, _ = guard_agent
    _configure(agent)
    begin(agent, REQUEST)
    begin(agent, new_request)
    assert agent._execution_target is None
    begin(agent, "Continue")
    assert agent._execution_target is None


def test_local_host_hint_uses_cached_machine_fact():
    from agent.prompt_builder import _local_host_hints
    from hermes_platform.host import facts
    with patch.object(facts.platform, "node", return_value="fixture-mac.local"):
        facts.clear_caches()
        assert "Terminal machine: fixture-mac.local" in "\n".join(_local_host_hints())
    facts.clear_caches()
