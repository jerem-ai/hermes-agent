"""Discovery cycles longer than four terminal reads must still be bounded."""
import json
import pytest
from agent.tool_guardrails import ToolCallGuardrailConfig, ToolCallGuardrailController


def observe_cycle(controller, lap, *, changed=False, poll=False):
    notices = []
    for i in range(6):
        name = 'process_manage' if poll else 'terminal'
        args = {'command': f'ssh macbook-pro cat source-{i}.py'}
        output = json.dumps({'output': f'source {i} revision {lap if changed else 0}', 'exit_code': 0})
        controller.after_call(name, args, output, failed=False)
        notices.append(controller.observe_call(name, args, output, tool_call_id=f'{lap}-{i}'))
    return notices


def test_long_terminal_discovery_cycle_warns_then_halts():
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(hard_stop_enabled=True))
    assert all(n.notice is None for n in observe_cycle(controller, 0))
    observe_cycle(controller, 1)
    assert any(n.notice for n in observe_cycle(controller, 2))
    observe_cycle(controller, 3)
    observe_cycle(controller, 4)
    assert controller.halt_decision.code == 'identical_cycle_halt'
    assert controller.halt_decision.count == 5


@pytest.mark.parametrize('changed,poll', [(True, False), (False, True)])
def test_changed_evidence_and_legitimate_poll_cycles_continue(changed, poll):
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(hard_stop_enabled=True))
    for lap in range(6):
        assert all(n.notice is None for n in observe_cycle(controller, lap, changed=changed, poll=poll))
    assert controller.halt_decision is None


def test_long_discovery_cycle_halts_real_turn_before_more_provider_calls():
    from unittest.mock import patch
    from tests.agent.test_tool_call_guardrail_runtime import _make_agent, _mock_response, _mock_tool_call
    agent = _make_agent('terminal', max_iterations=45, platform='bluebubbles')
    agent._disable_streaming = True
    responses = [_mock_response(content='', finish_reason='tool_calls', tool_calls=[
        _mock_tool_call('terminal', json.dumps({'command': f'cat source-{n % 6}.py'}), f'read-{n}')
    ]) for n in range(36)]
    responses.append(_mock_response(content='Complete.'))
    agent.client.chat.completions.create.side_effect = responses
    def read(name, args, task_id, **kwargs):
        return json.dumps({'output': args['command'], 'exit_code': 0})
    with (patch('model_tools.handle_function_call', side_effect=read) as execute,
          patch.object(agent, '_save_trajectory'), patch.object(agent, '_cleanup_task_resources')):
        result = agent.run_conversation('Inspect the source, make the repair and verify it.')
    assert result['turn_exit_reason'] == 'guardrail_halt'
    assert execute.call_count <= 30
    assert agent.client.chat.completions.create.call_count <= 30
    assert result['completed'] is False
