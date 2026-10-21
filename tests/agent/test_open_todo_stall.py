"""Open-todo stall guard: wording-independent continuation from the model's own todo list."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.completion_announcements import open_todo_stall, unfulfilled_action_announcement
from tests.agent.test_completion_announcement_recovery import DONE, _final, _run, _tool, guard_agent  # noqa: F401
from tools.todo_tool import TodoStore

PAIR = [{"id": "pair", "content": "Pair the Bluetooth controller with the Apple TV", "status": "pending"},
        {"id": "rom", "content": "Copy the ROM into RetroArch", "status": "completed"}]
MOVING_ON = "Moving on to the controller pairing."


def _stub(items, tools=("terminal",)):
    store = TodoStore()
    if items:
        store.write(items)
    return SimpleNamespace(_todo_store=store, valid_tool_names=set(tools))


def test_unlisted_wording_escapes_the_phrase_guard():
    # The defect this guard closes: the phrase list does not know this tail.
    assert unfulfilled_action_announcement(MOVING_ON, "Set up the ROM hub.") is False
    assert open_todo_stall(_stub(PAIR), MOVING_ON, "Set up the ROM hub.", tool_rows=2) is True


@pytest.mark.parametrize("text,user_request,tool_rows,items", [
    ("Should I pair the controller now?", "Set up the ROM hub.", 2, PAIR),
    ("Pairing is blocked: the controller needs you to hold its pairing button.", "Set up the ROM hub.", 2, PAIR),
    ("I can pair it if you want.", "Set up the ROM hub.", 2, PAIR),
    ("It is 10 pm.", "What time is it?", 0, PAIR),
    ("All set. I'll check it again tomorrow.", "Set up the ROM hub.", 3,
     [dict(i, status="completed") for i in PAIR]),
    ("Done.", "Set up the ROM hub.", 2, []),
    ("Here is the plan.", "[discuss] how would you set up the ROM hub?", 2, PAIR),
], ids=["question", "blocker", "offer", "no-tool-work", "all-complete", "no-todos", "discussion"])
def test_valid_stops_stay_final(text, user_request, tool_rows, items):
    assert open_todo_stall(_stub(items), text, user_request, tool_rows) is False


def test_no_tools_offered_never_continues():
    assert open_todo_stall(_stub(PAIR, tools=()), MOVING_ON, "Set up the ROM hub.", tool_rows=2) is False


def test_full_turn_continues_then_finishes(guard_agent):
    agent, db = guard_agent
    agent._todo_store.write(PAIR)

    def finish_then_tool():
        agent._todo_store.write([dict(i, status="completed") for i in PAIR])
        return _tool(call_id="continued_1")

    result, effects = _run(agent, "Set up the ROM hub.", [_tool(), _final(MOVING_ON), finish_then_tool, _final(DONE)])
    assert result["completed"] is True
    assert result["final_response"] == DONE
    assert effects == 2
    requests = [c.kwargs for c in agent.client.chat.completions.create.call_args_list]
    nudges = [m for m in requests[2]["messages"] if m.get("role") == "user" and "todo list still has open items" in str(m.get("content"))]
    assert nudges and "Pair the Bluetooth controller" in nudges[0]["content"]
    assert all(r.get("tools") == requests[0].get("tools") for r in requests)


def test_full_turn_question_is_delivered_without_nudge(guard_agent):
    agent, db = guard_agent
    agent._todo_store.write(PAIR)
    question = "The ROM is copied. Should I pair the controller now?"
    result, effects = _run(agent, "Set up the ROM hub.", [_tool(), _final(question)])
    assert result["completed"] is True
    assert result["final_response"] == question
    assert agent.client.chat.completions.create.call_count == 2
    assert effects == 1


def test_full_turn_exhaustion_delivers_answer_without_failure(guard_agent):
    agent, db = guard_agent
    agent._todo_store.write(PAIR)
    # One nudge per tool-work window: the nudge row closes the window, so a repeated stop is delivered.
    result, effects = _run(agent, "Set up the ROM hub.", [_tool(), _final(MOVING_ON), _final(MOVING_ON)])
    assert agent.client.chat.completions.create.call_count == 3
    assert effects == 1
    assert result["completed"] is True
    assert not result.get("failed")
    assert result["final_response"] == MOVING_ON
    durable = db.get_messages_as_conversation(agent.session_id)
    assert any(m.get("role") == "assistant" and m.get("content") == MOVING_ON for m in durable)
