"""Reclaim post-handoff work when the recency allowance exceeds request headroom."""
import logging
from typing import Any

logger = logging.getLogger(__name__)


def scan_pressure_window(compressor: Any, messages: list[dict], start: int, end: int,
                         request_tokens: int | None):
    """Keep ordinary tail protection; retry an empty handoff window once under pressure.

    The normal recency floor excludes fixed system/tool overhead. If it consumes the entire
    post-handoff body, reduce its soft ceiling by the request's excess over the trigger.
    The existing boundary selector still owns user, media and atomic tool-group protection.
    """
    if getattr(compressor, "tail_mode", "lean") == "lean":
        messages = compressor._demote_stale_tail_tools(messages, end)
    scan = compressor._scan_window_handoffs(messages, start, end, messages[start:end])
    if (scan.turns_to_summarize or not isinstance(request_tokens, int)
            or request_tokens <= compressor.threshold_tokens):
        return messages, end, scan
    # _find_tail_cut_by_tokens permits a 1.5x soft overrun. Price the reduced allowance
    # against that ceiling, or the old tail can still fit and pin all newer work.
    excess = request_tokens - compressor.threshold_tokens
    tail_cost = sum(map(compressor._tail_row_pricer(messages), range(end, len(messages))))
    # An almost-at-trigger request can need only a few tokens, less than an indivisible
    # row. Make at least half the optional tail eligible so anchors cannot undo that cut.
    allowance = min(tail_cost - excess, tail_cost / 2, compressor.tail_token_budget)
    budget = max(1, int(allowance / 1.5))
    _, retry_end = compressor._compress_window(messages, token_budget=budget)
    if retry_end <= end:
        return messages, end, scan
    retry_scan = compressor._scan_window_handoffs(messages, start, retry_end, messages[start:retry_end])
    if not retry_scan.turns_to_summarize:
        return messages, end, scan
    # Abort must restore the state from before either scan rehydrated the saved handoff.
    retry_scan.previous_summary_before = scan.previous_summary_before
    retry_scan.has_user_turn_before = scan.has_user_turn_before
    logger.info("Post-handoff pressure cut: request=%d trigger=%d tail_budget=%d->%d end=%d->%d",
                request_tokens, compressor.threshold_tokens, compressor.tail_token_budget, budget, end, retry_end)
    return messages, retry_end, retry_scan
