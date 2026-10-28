"""Keep a pending request alive while a local gateway has not admitted it.

Only structured, loopback admission refusals qualify. Transport errors, safety
locks, mixed route failures and requests with delivered text keep normal recovery.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urlsplit


def local_admission_pending(agent: Any, error: Any) -> bool:
    """True only when every eligible local route refused admission as busy."""
    if urlsplit(str(getattr(agent, "base_url", "") or "")).hostname not in {
        "localhost", "127.0.0.1", "::1",
    }:
        return False
    if getattr(error, "status_code", None) not in {502, 503}:
        return False
    body = getattr(error, "body", None)
    if not isinstance(body, dict):
        return False
    detail = body.get("error", body)
    if not isinstance(detail, dict) or detail.get("code") != "all_routes_failed":
        return False
    failures = detail.get("failures")
    if not isinstance(failures, list) or not failures or not all(
        isinstance(failure, dict) and failure.get("reason") == "local_busy"
        for failure in failures
    ):
        return False
    streamed = getattr(agent, "_current_streamed_assistant_text", "") or ""
    return not agent._has_content_after_think_block(streamed)


def wait_for_local_admission(
    agent: Any, error: Any, retry: Any, *, messages: Any,
    conversation_history: Any, api_call_count: int,
) -> dict[str, Any] | None:
    """Wait cancellably without spending an inference retry or replaying prior tools."""
    if not local_admission_pending(agent, error):
        return None
    from agent.turn_recovery import interruptible_backoff_sleep

    hint = "press Esc to stop" if str(getattr(agent, "platform", "")).lower() in {"cli", "tui", "desktop"} else "cancel this turn to stop waiting"
    notice = f"Waiting for the local model to become available. The pending step will resume automatically; {hint}."
    # Older runtimes expose these through _vprint and thinking_callback.
    emit_status = getattr(agent, "_emit_diagnostic_status", None)
    if emit_status is None:
        agent._vprint(notice, force=True)
    else:
        emit_status(notice)
    emit_wait = getattr(agent, "_emit_diagnostic_wait", None)
    if emit_wait is not None:
        emit_wait(notice)
    elif agent.thinking_callback:
        agent.thinking_callback(notice)
    agent._touch_activity("Waiting for local model admission")
    interrupted = interruptible_backoff_sleep(
        agent, 15.0, retry, messages=messages, conversation_history=conversation_history,
        api_call_count=api_call_count,
        abort_message="Stopped while waiting for the local model.",
        interrupt_text="Stopped while waiting for the local model. Review prior tool results before continuing.",
        activity_label="Waiting for local model admission",
    )
    if interrupted is not None:
        return {"action": "return", "result": interrupted}
    return {"action": "break" if retry.restart_with_redirected_messages else "continue"}
