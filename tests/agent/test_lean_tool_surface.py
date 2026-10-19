"""Deferred capability discovery preserves schemas and full-turn dispatch boundaries."""
import json
from unittest.mock import patch

import pytest

from tests.agent.test_completion_announcement_recovery import _final, _run
from tests.agent.test_run_agent import _mock_response, _mock_tool_call
from tools import tool_search as ts
from tools.registry import registry

pytest_plugins = ['tests.agent.test_completion_announcement_recovery']


@pytest.fixture
def surface(monkeypatch):
    browser = {'type':'function','function':{'name':'browser_navigate','description':'Browser navigate to a URL. ' * 160,
               'parameters':{'type':'object','properties':{'url':{'type':'string'}},'required':['url']}}}
    terminal = {'type':'function','function':{'name':'terminal','description':'Run a terminal command.',
                'parameters':{'type':'object','properties':{'command':{'type':'string'}},'required':['command']}}}
    registry.register(name='browser_navigate', schema=browser['function'], handler=lambda *_a,**_k:'{}',toolset='browser')
    registry.register(name='terminal', schema=terminal['function'], handler=lambda *_a,**_k:'{}',toolset='terminal')
    cfg = ts.ToolSearchConfig.from_raw({'enabled':'on','listing_max_tokens':1000,'defer':['browser_navigate']})
    monkeypatch.setattr(ts,'load_config',lambda:cfg)
    monkeypatch.setattr(ts,'load_config_readonly',lambda:cfg)
    return [terminal,browser],cfg


def test_reduction_retains_exact_discoverable_schema(surface):
    full,cfg = surface
    small = ts.assemble_tool_defs(full,context_length=65536,config=cfg)
    names={d['function']['name'] for d in small.tool_defs}
    assert 'terminal' in names and 'browser_navigate' not in names
    assert len(json.dumps(small.tool_defs)) < len(json.dumps(full))
    found=json.loads(ts.dispatch_tool_search({'queries':['browser navigate']},current_tool_defs=full,config=cfg))
    assert 'browser_navigate' in found['tools']
    described=json.loads(ts.dispatch_tool_describe({'names':['browser_navigate']},current_tool_defs=full,config=cfg))
    assert described['tools']['browser_navigate']['parameters']==full[1]['function']['parameters']


def _turn(agent,surface,args):
    full,cfg=surface
    compact=ts.assemble_tool_defs(full,context_length=65536,config=cfg).tool_defs
    agent.tools=compact
    agent.valid_tool_names={d['function']['name'] for d in compact}
    response=_mock_response(content='',finish_reason='tool_calls',tool_calls=[
        _mock_tool_call(name='tool_call',arguments=json.dumps({'calls':[{'name':'browser_navigate','arguments':args}]}),call_id='deferred-browser')])
    with patch('model_tools.get_tool_definitions',return_value=full):
        return _run(agent,'Inspect the requested browser page.',[response,_final('The requested page was inspected.')],
                    tool_result='{"output":"fixture page","exit_code":0}')


def test_unusual_correct_deferred_call_persists_and_keeps_cached_prefix(guard_agent,surface):
    agent,db=guard_agent
    result,effects=_turn(agent,surface,{'url':'https://example.test/雪?q=a%20b&empty='})
    assert effects==1 and result['completed']
    assert db.get_messages_as_conversation(agent.session_id)[-1]['content']==result['final_response']
    requests=agent.client.chat.completions.create.call_args_list
    assert requests[0].kwargs['tools']==requests[1].kwargs['tools']
    assert [m for m in requests[0].kwargs['messages'] if m['role']=='system']==[
        m for m in requests[1].kwargs['messages'] if m['role']=='system']


def test_deferred_missing_required_field_is_refused_before_dispatch(guard_agent,surface):
    agent,_=guard_agent
    result,effects=_turn(agent,surface,{})
    assert effects==0
    assert any('url' in str(m.get('content')) for m in result['messages'] if m.get('role')=='tool')


def test_existing_agent_refreshes_discovery_scope_after_deferral_change(surface,monkeypatch):
    from types import SimpleNamespace
    from agent.tool_executor import _tool_search_scoped_names
    full,cfg=surface
    current=[ts.ToolSearchConfig.from_raw({'defer':[]})]
    monkeypatch.setattr(ts,'load_config_readonly',lambda:current[0])
    monkeypatch.setattr(ts,'load_config',lambda:current[0])
    agent=SimpleNamespace(enabled_toolsets=None,disabled_toolsets=None)
    with patch('model_tools.get_tool_definitions',return_value=full):
        assert 'browser_navigate' not in _tool_search_scoped_names(agent)
        current[0]=cfg
        refreshed=_tool_search_scoped_names(agent)
        assert 'browser_navigate' in refreshed
        assert 'terminal' not in refreshed
