"""Keep a production-sized skill catalog discoverable without paying descriptions every turn."""
import json
from types import SimpleNamespace

import pytest

from agent import prompt_builder as pb
from agent.skills_index_delta import index_entries, stage_skills_index_note

pytest_plugins = ['tests.agent.test_completion_announcement_recovery']


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pb, 'get_disabled_skill_names', lambda *_: set())
    pb.clear_skills_system_prompt_cache(clear_snapshot=True)
    for index in range(145):
        name = 'inspect-雪' if index == 144 else f'catalog-{index:03d}'
        path = tmp_path / 'skills' / f'category-{index % 10}' / name / 'SKILL.md'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'---\nname: {name}\ndescription: ' + 'Useful operating instructions with exact tool evidence. ' + '\n---\nFull instructions 雪.\n')
    return tmp_path


def config(home, mode):
    (home / 'config.yaml').write_text(json.dumps({'skills': {'index_mode': mode}}))


def test_large_catalog_reduces_prompt_without_losing_names(catalog):
    config(catalog, 'full')
    full = pb.build_skills_system_prompt()
    config(catalog, 'names')
    compact = pb.build_skills_system_prompt()
    assert len(index_entries(full)) == 145
    assert set(index_entries(compact)) == set(index_entries(full))
    assert len(compact) < len(full) * .55
    assert 'skills_list' in compact and 'outside the current coding' not in compact


def test_unusual_skill_keeps_full_description_and_body_available(catalog):
    from tools.skills_tool import skill_view, skills_list
    config(catalog, 'names')
    assert 'inspect-雪' in index_entries(pb.build_skills_system_prompt())
    listed = json.loads(skills_list(category='category-4'))
    entry = next(s for s in listed['skills'] if s['name'] == 'inspect-雪')
    assert 'Useful operating instructions' in entry['description']
    viewed = json.loads(skill_view('inspect-雪'))
    assert viewed['success'] and 'Full instructions 雪.' in viewed['content']


def test_names_mode_is_profile_owned_and_invalid_value_keeps_full(catalog, tmp_path):
    config(catalog, 'names')
    compact = pb.build_skills_system_prompt()
    other = tmp_path / 'other-profile'
    other.mkdir()
    (other / 'skills').symlink_to(catalog / 'skills', target_is_directory=True)
    config(other, 'not-a-mode')
    full = pb.build_skills_system_prompt(skills_dir_override=other / 'skills')
    assert len(full) > len(compact)
    assert 'Useful operating instructions' in full
    assert pb.build_skills_system_prompt() == compact


def test_late_skill_change_keeps_cached_prefix_and_long_history(catalog):
    config(catalog, 'names')
    prompt = 'Identity\n' + pb.build_skills_system_prompt()
    history = [{'role': role, 'content': 'source evidence 雪 ' * 450}
               for _ in range(85) for role in ['user', 'assistant']]
    before = [dict(row) for row in history]
    path = catalog / 'skills' / 'category-0' / 'late-skill' / 'SKILL.md'
    path.parent.mkdir(parents=True)
    path.write_text('---\nname: late-skill\ndescription: Late instructions.\n---\nBody.\n')
    agent = SimpleNamespace(provider='custom', api_mode='chat_completions', platform='bluebubbles',
                            valid_tool_names={'skills_list','skill_view'}, session_id='compact-fixture',
                            _session_db=SimpleNamespace(db_path=str(catalog / 'state.db')))
    assert stage_skills_index_note(agent, prompt, history)
    assert 'late-skill' in agent._skills_index_note
    assert history == before
    assert 'late-skill' not in prompt


def test_compact_index_does_not_advertise_an_unavailable_listing_tool(catalog):
    config(catalog, 'names')
    prompt = pb.build_skills_system_prompt(available_tools={'skill_view'})
    assert 'Use skills_list(category)' not in prompt
    assert 'inspect-雪' in index_entries(prompt)


def test_native_large_context_compaction_retains_compact_catalog(catalog, guard_agent):
    from tests.agent.test_compression_abort_state_reset import _InPlaceSuccessCompressor
    agent, db = guard_agent
    config(catalog, 'names')
    agent.valid_tool_names.update({'skills_list', 'skill_view'})
    agent._cached_system_prompt = 'Identity\n' + pb.build_skills_system_prompt()
    db.ensure_session(agent.session_id, source='bluebubbles')
    messages = [{'role': role, 'content': 'source evidence 雪 ' * 450}
                for _ in range(85) for role in ['user', 'assistant']]
    assert len(messages) == 170 and sum(len(m['content']) for m in messages) > 240000
    agent.context_compressor = _InPlaceSuccessCompressor()
    agent._compression_feasibility_checked = True
    agent.compression_in_place = True
    agent._compress_context(messages, agent._cached_system_prompt, approx_tokens=60000, force=True)
    entries = index_entries(agent._cached_system_prompt)
    assert len(entries) == 145 and 'inspect-雪' in entries
    assert '[names only]' in agent._cached_system_prompt
    assert agent.client.chat.completions.create.call_count == 0
