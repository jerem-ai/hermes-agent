"""Main and compression use their own reasoning controls on the local custom wire."""
from unittest.mock import patch

from tests.agent import test_completion_announcement_recovery as helpers
from hermes_constants import resolve_reasoning_config

guard_agent = helpers.guard_agent


def test_main_request_keeps_explicit_xhigh(guard_agent):
    agent, _db = guard_agent
    agent.reasoning_config = resolve_reasoning_config({'agent': {'reasoning_effort': 'xhigh'}}, agent.model)
    result, _effects = helpers._run(agent, 'Report the saved checks.', [helpers._final('The checks passed.')])
    assert result['completed'] is True
    assert agent.client.chat.completions.create.call_args.kwargs['reasoning_effort'] == 'xhigh'


def test_compression_custom_wire_disables_thinking():
    from agent.auxiliary_client import _get_task_extra_body
    from providers import get_provider_profile
    with patch('agent.auxiliary_client._get_auxiliary_task_config', return_value={'reasoning_effort': 'none'}):
        body = _get_task_extra_body('compression')
    assert body['reasoning'] == {'enabled': False}
    profile = get_provider_profile('custom')
    _extra, top = profile.build_api_kwargs_extras(model='leyline/tars', reasoning_config=body['reasoning'])
    assert top['reasoning_effort'] == 'none'
