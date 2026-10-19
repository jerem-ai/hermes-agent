"""Finish an oversized blocked recovery without sending or resetting its history."""
from __future__ import annotations

import json
import logging
from typing import Any

from agent.message_metadata import append_message
from agent.message_sanitization import _sanitize_surrogates


def _pending_action(messages: list[dict], user_message: Any) -> str:
    for message in reversed(messages):
        if message.get('role') == 'assistant' and isinstance(message.get('content'), str):
            if message['content'].strip():
                return _sanitize_surrogates(message['content'].strip()[-1200:])
    return _sanitize_surrogates(str(user_message)[-1200:])


def _source_location(messages: list[dict]) -> str:
    for message in reversed(messages):
        for call in reversed(message.get('tool_calls') or []):
            function = call.get('function') or {}
            try:
                args = json.loads(function.get('arguments') or '{}')
            except (ValueError, TypeError):
                continue
            if isinstance(args, dict):
                for key in ('path', 'cwd'):
                    value = args.get(key)
                    if isinstance(value, str) and value:
                        return _sanitize_surrogates(value[:600])
    return 'See the current request and its latest tool results.'


def preserve_blocked_recovery(agent: Any, verdict: Any, pressure: int, user_message: Any) -> Any:
    """Reuse the existing window predicate; fitting or ordinary unblocked sends are unchanged."""
    from agent.conversation_compression import request_exceeds_model_window
    from agent.turn_context_compaction import _blocked_compress_reason, _refund_api_call

    if (verdict.action != 'fallthrough' or not agent.compression_enabled
            or request_exceeds_model_window(agent, pressure) is not True):
        return verdict
    reason = _blocked_compress_reason(agent.context_compressor, pressure, verdict.compression_attempts)
    if not isinstance(reason, str) or not reason:
        return verdict
    from agent.turn_final_response import _queued_user_control

    if _queued_user_control(agent):
        verdict.api_call_count = _refund_api_call(agent, verdict.api_call_count)
        verdict.action = "continue"
        return verdict
    checkpoint = {
        'goal': _sanitize_surrogates(str(user_message)[:2400]),
        'pending_step': _pending_action(verdict.messages, user_message),
        'source_location': _source_location(verdict.messages),
        'constraints': 'Existing task approvals and resource limits still apply.',
    }
    text = (
        'Paused: estimated request size exceeds the configured context window and '
        f'compression is blocked ({reason}). No provider request was sent.\n\n'
        f'Task: {checkpoint["goal"]}\nPending: {checkpoint["pending_step"]}\n'
        f'Location: {checkpoint["source_location"]}\n{checkpoint["constraints"]}\n'
        'Run /compress after the blocker clears, or start /new with this task record.'
    )
    append_message(verdict.messages, {'role': 'assistant', 'content': text})
    try:
        saved = agent._persist_session(verdict.messages, verdict.conversation_history) is True
    except Exception:
        logging.getLogger(__name__).exception("Recovery checkpoint persistence failed")
        saved = False
    if not saved:
        text = ('Paused: could not save the recovery checkpoint. The task is unfinished.\n'
                f'Pending: {checkpoint["pending_step"]}\nLocation: {checkpoint["source_location"]}')
    calls = _refund_api_call(agent, verdict.api_call_count)
    agent._session_messages = verdict.messages
    verdict.action = 'return'
    verdict.result = {
        'final_response': text, 'messages': verdict.messages, 'completed': False,
        'failed': True, 'partial': True, 'compression_deferred': True,
        'checkpoint_saved': saved, 'checkpoint': checkpoint, 'api_calls': calls,
        'turn_exit_reason': 'compression_recovery_blocked', 'error': text,
        'failure_reason': 'context_overflow' if saved else 'session_persistence_failed',
        'failure_retryable': False, 'session_id': agent.session_id,
    }
    return verdict
