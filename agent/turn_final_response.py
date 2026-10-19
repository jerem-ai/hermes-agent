"""No-tool-call (final text) branch of the conversation turn loop: empty/think-only recovery,
intent-ack / stall-guard continuation, length-continuation joining, dropped-tool-call
re-prompt, scaffolding pop, stop gates, then the durable final flush. Extracted from
``run_conversation``; nothing here imports ``agent.conversation_loop`` at module level
(cycle) — loop-internal nudge constants resolve lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional

from agent.message_metadata import append_message
from agent.reasoning_promotion import answer_in_reasoning_capability
from agent.repetition_guard import STOP_PATH_MIN_CHARS, is_runaway_repetition
from agent.turn_failure_copy import (
    ANNOUNCEMENT_EXHAUSTED_REASON, announcement_exhausted_copy, stamp_failure,
)
from agent.turn_empty_response import recover_empty_response
from agent.turn_stop_gates import apply_stop_gates
from agent.turn_truncation import partial_result, repetition_copy

_REPETITION_STOPPED = repetition_copy(
    "before delivery",
    "so the repeated output was discarded.",
    "; refusing to return a",
)
logger = logging.getLogger("agent.conversation_loop")

# Ephemeral retry scaffolding rows popped before the final answer becomes durable.
_EPHEMERAL_SCAFFOLDING_FLAGS = (
    "_thinking_prefill", "_empty_recovery_synthetic", "_empty_terminal_sentinel",
    "_dropped_toolcall_nudge",
)


def _queued_user_control(agent: Any) -> Optional[str]:
    """Live user control that owns the next step: "redirect", "interrupt" or "steer" (None when idle)."""
    if agent._has_pending_redirect() is True:
        return "redirect"
    if getattr(agent, "_interrupt_requested", False) is True:
        return "interrupt"
    from agent.interrupt_control import _ic_lock, _ic_slot
    with _ic_lock(agent, "_pending_steer_lock"):
        pending = _ic_slot(agent, "_pending_steer_lock", "_pending_steer")
    return "steer" if isinstance(pending, str) and pending.strip() else None



def _continuation_choice(agent, final_response, promoted, messages, user_message, continuations):
    """Choose a bounded continuation; preserve valid quoted, paused and completed answers."""
    from agent.agent_runtime_helpers import (
        intent_ack_continuation_mode, looks_like_degenerate_final, promoted_reasoning_announces_action,
        tool_results_this_turn,
    )
    from agent.completion_announcements import (
        latest_execution_request_text, unfulfilled_action_announcement,
        request_forbids_action, has_user_condition_or_offer,
    )
    _ack_mode = intent_ack_continuation_mode(agent)
    # Said-continue-but-stopped guard: no tool calls but the short reply TAILS with an
    # announced next action. Reuses the SAME bounded continuation counter (max 2 per turn).
    # Promoted reasoning gets the broader first-person-plan tail detector: with tools offered
    # and zero tool calls, chain-of-thought ending on "Let me batch the terminal calls..." is a
    # stalled model, and returning it as the answer aborts the tool loop while reporting
    # "complete" (#111761). Same cap, so a model that never acts still ends after 2 nudges.
    _stall_text = agent._strip_think_blocks(final_response or "")
    _stall_request = latest_execution_request_text(messages)
    # Eligibility is judged independently of the counter (and of the ack setting): a stalled
    # candidate past the cap ends the turn as an explicit failure instead of passing as an answer.
    _stall_candidate = bool(
        bool(getattr(agent, "_stall_guards", True))
        and agent.valid_tool_names
        and (
            unfulfilled_action_announcement(_stall_text, _stall_request)
            or (
                bool(promoted)
                and not request_forbids_action(_stall_request)
                and not has_user_condition_or_offer(_stall_text)
                and promoted_reasoning_announces_action(_stall_text)
            )
        )
    )
    _stall_continue_intent = _stall_candidate and continuations < 2
    _stall_exhausted = _stall_candidate and not _stall_continue_intent
    # Degenerate-final guard (#103483): the turn did real tool work and then stopped on a
    # fragment. Same scope knob and the SAME bounded counter as the ack continuation; the nudge
    # row itself closes the tool-work window, so a second fragment ends the turn as the answer.
    _tool_rows = tool_results_this_turn(messages)
    _degenerate_final = (
        bool(getattr(agent, "_stall_guards", True))
        and _ack_mode != "off"
        and continuations < 2
        and _tool_rows > 0
        and looks_like_degenerate_final(_stall_text, user_message=user_message)
    )
    # Precedence: an announced next action outranks the fragment shape; the codex ack is last.
    if _stall_continue_intent:
        _continuation_kind = "stall"
    elif _degenerate_final:
        _continuation_kind = "degenerate"
    elif (
        _ack_mode != "off"
        and agent.valid_tool_names
        and continuations < 2
        and agent._looks_like_codex_intermediate_ack(
            user_message=user_message, assistant_content=final_response, messages=messages,
            require_workspace=(_ack_mode == "codex_only"),
        )
    ):
        _continuation_kind = "ack"
    else:
        _continuation_kind = None
    return _continuation_kind, _stall_exhausted, _stall_text, _tool_rows


def _apply_pending_control(agent, messages, assistant_message, promoted, final_response):
    """Give live user control precedence over any continuation or exhaustion effect."""
    control = _queued_user_control(agent)
    steer = agent._drain_pending_steer() if control == "steer" else None
    if control == "steer" and not steer:
        return False
    if control == "interrupt":
        return True
    if control not in ("redirect", "steer"):
        return False
    interim = agent._build_assistant_message(assistant_message, "incomplete")
    if promoted:
        interim["api_content"] = final_response
    append_message(messages, interim)
    agent._emit_interim_assistant_message(interim)
    if steer:
        from agent.conversation_loop import _apply_active_turn_redirect
        _apply_active_turn_redirect(agent, messages, steer)
    return True

@dataclass
class FinalResponseVerdict:
    """``action``: ``"break"`` (turn ends with ``final_response``), ``"continue"`` (a
    continuation/re-prompt/stop-gate asked for another API call) or ``"return"``
    (``result`` is the turn's result dict). The other fields are the loop locals rebound."""

    action: str
    active_system_prompt: Any
    final_response: Any
    _turn_exit_reason: Any
    _preflight_compression_blocked: Any
    codex_ack_continuations: Any
    truncated_response_parts: Any
    length_continue_retries: Any
    _pending_verification_response: Any
    _pending_verification_response_previewed: Any
    api_call_count: int
    result: Optional[dict[str, Any]] = None


def finish_text_response(
    agent: Any, *, assistant_message: Any, response: Any, finish_reason: Any, messages: Any,
    api_messages: Any, conversation_history: Any, api_call_count: Any, user_message: Any,
    active_system_prompt: Any, final_response: Any, _turn_exit_reason: Any,
    _preflight_compression_blocked: Any, codex_ack_continuations: Any,
    truncated_response_parts: Any, length_continue_retries: Any,
    _pending_verification_response: Any, _pending_verification_response_previewed: Any,
    effective_task_id: Any,
) -> FinalResponseVerdict:
    """Finish (or defer) a text-only assistant response in the original guard order. Every
    continuation path sets ``final_response = None`` so an acknowledgment never suppresses
    iteration-limit summarization; the final message is appended and flushed only after the
    stop gates accept it."""
    from agent.conversation_loop import (
        _CODEX_ACK_CONTINUATION_NUDGE, _DEGENERATE_FINAL_NUDGE, _DROPPED_TOOLCALL_NUDGE_CONTENT,
        _join_truncated_parts
    )

    def _verdict(action: str, result: Optional[dict[str, Any]] = None) -> FinalResponseVerdict:
        return FinalResponseVerdict(
            action=action, active_system_prompt=active_system_prompt, final_response=final_response,
            _turn_exit_reason=_turn_exit_reason,
            _preflight_compression_blocked=_preflight_compression_blocked,
            codex_ack_continuations=codex_ack_continuations,
            truncated_response_parts=truncated_response_parts,
            length_continue_retries=length_continue_retries,
            _pending_verification_response=_pending_verification_response,
            _pending_verification_response_previewed=_pending_verification_response_previewed,
            api_call_count=api_call_count,
            result=result,
        )

    # Reasoning-only clean stop on a trusted route (elsewhere reasoning is private and takes the
    # ladder): some parsers (vLLM nemotron_v3 past ~500K prompt tokens) file the whole answer as
    # reasoning when the closing delimiter is missing; ``stop`` means generation is complete, so
    # promote it BEFORE the ladder. ``length`` (cut off mid-thought) stays on the continuation path.
    # The promoted text is RETURNED as the answer but never written into the assistant
    # row's ``content``: chain-of-thought stored as ordinary content is indistinguishable
    # from a real reply on every history surface (#111761). The row keeps ``content``
    # empty with the text in its reasoning fields and carries the promoted text as the
    # ``api_content`` sidecar, so the next turn still replays it byte-identically.
    # Anthropic thinking (signed ``thinking`` block, or a plugin's ``*.native_assistant`` carrier
    # of native Claude turns) is a summary written by a separate model, never the answer: it
    # takes the empty-response continuation below instead.
    _content = assistant_message.content
    _promoted = None
    if (
        finish_reason == "stop"
        and not assistant_message.tool_calls
        and (_content is None or (isinstance(_content, str) and not _content.strip()))
        and not any(
            isinstance(d, dict) and (
                (d.get("type") in ("thinking", "redacted_thinking") and (d.get("signature") or d.get("data")))
                or str(d.get("type") or "").endswith(".native_assistant")
            )
            for d in getattr(assistant_message, "reasoning_details", None) or ()
        )
        and answer_in_reasoning_capability(agent)
    ):
        _promoted = agent._extract_reasoning(assistant_message) or None
        if _promoted:
            # WARNING, not INFO: a model that keeps ending turns this way is stalled
            # (planning monologue, zero tool calls) while the turn reports "complete".
            logger.warning(
                "Reasoning-only clean stop (%d chars) — returning the reasoning as the final "
                "response (model=%s provider=%s api_calls=%d tool_turns=%d)",
                len(_promoted), agent.model, agent.provider, api_call_count,
                sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant" and m.get("tool_calls")),
            )
    final_response = _promoted or assistant_message.content or ""
    # Unmute: _mute_post_response from a housekeeping tool turn must not silence
    # empty-response warnings on the final response path.
    agent._mute_post_response = False

    # Think-block-only / empty content: recovery path.
    if not agent._has_content_after_think_block(final_response):
        _ev = recover_empty_response(
            agent, assistant_message, response, finish_reason, final_response=final_response,
            messages=messages, api_messages=api_messages, conversation_history=conversation_history,
            active_system_prompt=active_system_prompt, api_call_count=api_call_count,
            turn_exit_reason=_turn_exit_reason,
            preflight_compression_blocked=_preflight_compression_blocked,
        )
        final_response = _ev.final_response
        _turn_exit_reason = _ev.turn_exit_reason
        active_system_prompt = _ev.active_system_prompt
        _preflight_compression_blocked = _ev.preflight_compression_blocked
        api_call_count = _ev.api_call_count
        if _ev.action == "return":
            return _verdict("return", _ev.result)
        if _ev.action == "break":
            return _verdict("break")
        return _verdict("continue")

    agent._empty_content_retries = 0
    agent._thinking_prefill_retries = 0
    # Surface the one-shot fallback switch notice before dropping the retry buffer so a
    # provider/model switch stays visible on success.
    agent._emit_pending_fallback_notice()
    agent._clear_status_buffer()

    # Defensive: repair malformed role-alternation before API call. Catches cases where the history got
    # wedged into a ``tool → user`` or ``user → user`` tail (e.g. after empty- response scaffolding was
    # stripped and a new user message landed after an orphan tool result). Most providers return empty
    # content on malformed sequences, which would otherwise retrigger the empty-retry loop indefinitely.
    # repair_message_sequence_with_cursor also recomputes the SessionDB flush cursor (_last_flushed_db_idx)
    # when repair compacts the list, so the turn-end flush doesn't skip the assistant/tool chain (#44837).
    # One-time repeated-heal escalation notice (#96870): if the sanitizer above just crossed the per-session
    # heal threshold, deliver the queued notice through the status/warning callback — the normal out-of-band
    # delivery channel (gateway status message / CLI print). NEVER appended to messages/api_messages:
    # conversation context and the cached prompt prefix stay byte-identical.
    _continuation_kind, _stall_exhausted, _stall_text, _tool_rows = _continuation_choice(
        agent, final_response, _promoted, messages, user_message, codex_ack_continuations,
    )
    if (_continuation_kind or _stall_exhausted) and _apply_pending_control(
        agent, messages, assistant_message, _promoted, final_response,
    ):
        agent._session_messages = messages
        final_response = None
        return _verdict("continue")

    if _stall_exhausted:
        logger.warning(
            "Stall guard exhausted: %d-char announced action still not taken after %d nudge(s) — "
            "ending the turn as incomplete", len(_stall_text), codex_ack_continuations,
        )
        # Neutral about save status: finalize_turn words the checkpoint only after the flush outcome is known.
        # Promoted reasoning keeps the older 240-char crop so no more private reasoning is exposed.
        _exhausted_text = announcement_exhausted_copy(
            _stall_text[-240:] if _promoted else _stall_text, saved=None,
        )
        # Ordinary transcript row; finalize_turn persists it (and reports a failed save) and owns the result.
        _exhausted_msg = agent._build_assistant_message(assistant_message, finish_reason)
        _exhausted_msg["content"] = _exhausted_text
        _exhausted_msg.pop("api_content", None)
        append_message(messages, _exhausted_msg)
        agent._session_messages = messages
        final_response = _exhausted_text
        _turn_exit_reason = ANNOUNCEMENT_EXHAUSTED_REASON
        return _verdict("break")

    if _continuation_kind:
        if _continuation_kind == "stall":
            logger.info(
                "Stall guard: turn ending on trailing continue-"
                "intent with no tool calls — re-prompting to act "
                "(%d/2)", codex_ack_continuations + 1,
            )
        elif _continuation_kind == "degenerate":
            logger.warning(
                "Degenerate final: %d-char fragment %r ended the turn after %d tool result(s) — "
                "re-prompting (%d/2)", len(_stall_text), _stall_text[:40], _tool_rows,
                codex_ack_continuations + 1,
            )
        codex_ack_continuations += 1
        interim_msg = agent._build_assistant_message(assistant_message, "incomplete")
        if _promoted:
            # Same sidecar as the final row: the wire copy must carry the promoted text, not only
            # ``reasoning_content``, or the continuation replays an empty assistant turn.
            interim_msg["api_content"] = final_response
        append_message(messages, interim_msg)
        agent._emit_interim_assistant_message(interim_msg)
        append_message(messages, {
            "role": "user",
            "content": (
                _DEGENERATE_FINAL_NUDGE if _continuation_kind == "degenerate"
                else _CODEX_ACK_CONTINUATION_NUDGE
            ),
        })
        agent._session_messages = messages
        # An acknowledgment is non-final: its text must not suppress iteration-limit
        # summarization if the continuation exhausts budget.
        final_response = None
        return _verdict("continue")

    if truncated_response_parts:
        final_response = _join_truncated_parts([*truncated_response_parts, (final_response, False)])
        truncated_response_parts = []
        length_continue_retries = 0
        # The continuation recovered, so the fragments stay in the transcript.
        for _frag in messages:
            if isinstance(_frag, dict):
                _frag.pop("_length_continuation_fragment", None)
                _frag.pop("_length_continuation_nudge", None)

    final_response = agent._strip_think_blocks(final_response).strip()

    # A provider may end a degenerate loop normally with finish_reason="stop" instead of
    # exhausting its output cap (#100716). Check every completed visible text response before
    # any verify/kanban interim emission or durable transcript write.
    # Runaway scale and shape only: a completed answer the user asked to be repetitive is
    # delivered, unlike a length-truncated fragment that burned the whole budget.
    if (
        final_response
        and len(final_response) >= STOP_PATH_MIN_CHARS
        and is_runaway_repetition(final_response)
    ):
        line, user_response, error = _REPETITION_STOPPED
        agent._vprint(f"{agent.log_prefix}{line}", force=True, diagnostic=True)
        agent._cleanup_task_resources(effective_task_id)
        agent._persist_session(messages, conversation_history)
        return _verdict("return", stamp_failure(
            partial_result(messages, api_call_count, user_response, error), "truncated", True,
        ))

    final_msg = agent._build_assistant_message(assistant_message, finish_reason)
    if _promoted:
        # Replay sidecar only: ``content`` stays empty so the row is never mistaken for a
        # real reply; ``build_api_messages`` substitutes ``api_content`` on the wire.
        final_msg["api_content"] = final_response

    # Dropped tool-call recovery (copilot/Claude): finish_reason="tool_calls" with empty
    # tool_calls would end the turn unstarted; re-prompt (max 3 CONSECUTIVE stalls).
    if (
        finish_reason == "tool_calls"
        and not assistant_message.tool_calls
        and getattr(agent, "_dropped_toolcall_retries", 0) < 3
    ):
        agent._dropped_toolcall_retries = getattr(agent, "_dropped_toolcall_retries", 0) + 1
        logger.warning(
            "finish_reason=tool_calls with empty tool_calls array "
            "(narration only) — re-prompting to emit the call "
            "(retry %d/3, model=%s provider=%s)",
            agent._dropped_toolcall_retries, agent.model, agent.provider,
        )
        agent._emit_diagnostic_status(
            "↻ Model signaled a tool call but sent none — "
            f"re-prompting ({agent._dropped_toolcall_retries}/3)"
        )
        # Both halves of the re-prompt pair are ephemeral scaffolding: never persisted,
        # and the finalization pop strips an unanswered tail pair.
        final_msg["_dropped_toolcall_nudge"] = True
        append_message(messages, final_msg)
        append_message(messages, {
            "role": "user",
            "content": _DROPPED_TOOLCALL_NUDGE_CONTENT,
            "_dropped_toolcall_nudge": True,
        })
        agent._session_messages = messages
        final_response = None
        return _verdict("continue")

    # Genuine turn end (no dropped-tool-call mismatch): clear stall budget.
    agent._dropped_toolcall_retries = 0

    # Pop prefill / empty-retry scaffolding before the final response or
    # verification follow-up; it must not become durable transcript.
    while (
        messages
        and isinstance(messages[-1], dict)
        and any(messages[-1].get(flag) for flag in _EPHEMERAL_SCAFFOLDING_FLAGS)
    ):
        messages.pop()

    from agent.acceptance_contract import enforce_final
    final_response, _contract_blocked = enforce_final(agent, final_response, final_msg)

    _sg = apply_stop_gates(
        agent, final_msg, final_response=final_response, messages=messages,
        conversation_history=conversation_history,
        pending_verification_response=_pending_verification_response,
        pending_verification_response_previewed=_pending_verification_response_previewed,
    )
    _pending_verification_response = _sg.pending_verification_response
    _pending_verification_response_previewed = _sg.pending_verification_response_previewed
    if _sg.continue_turn:
        final_response = None
        return _verdict("continue")

    # Plugins rewrite the reply BEFORE it is appended and flushed: SQLite treats a non-blank
    # assistant row as settled, so a transform after this write would reach the user but never
    # the stored/replayed transcript (#44239). finalize_turn reads the recorded outcome; like
    # there, an interrupted turn keeps the raw text.
    from agent.turn_finalizer import apply_llm_output_transform
    _transformed = False
    if not getattr(agent, "_interrupt_requested", False):
        final_response, _transformed, _ = apply_llm_output_transform(
            agent, final_response, turn_id=getattr(agent, "_current_turn_id", "") or "", logger=logger,
        )
    from agent.acceptance_contract import reconcile_output
    final_response = reconcile_output(agent, final_response, final_msg, _promoted, _transformed)

    append_message(messages, final_msg)
    # Make the answer durable before leaving the loop (_DB_PERSISTED_MARKER keeps
    # _persist_session idempotent). Failure must NOT abort the turn: finalize retries.
    try:
        agent._flush_messages_to_session_db(messages, conversation_history)
    except Exception:
        logger.warning(
            "final text-turn flush failed (session=%s) — reply is "
            "not yet durable; relying on finalize_turn retry",
            getattr(agent, "session_id", None) or "none",
            exc_info=True,
        )

    _turn_exit_reason = f"text_response(finish_reason={finish_reason})"
    if not agent.quiet_mode:
        agent._safe_print(f"🎉 Conversation completed after {api_call_count} OpenAI-compatible API call(s)")
    return _verdict("break")
