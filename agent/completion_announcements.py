"""Detect unfinished execution announcements without treating valid answers as stalls."""
from __future__ import annotations

import re
from typing import Any
from agent.agent_runtime_helpers import (
    _TRAILING_CONTINUE_INTENT_MAX_CHARS, _TRAILING_CONTINUE_INTENT_RE,
)

# Announced-action tail for an ACTIVE execution request (stall guard): "Let me pull docs from there:",
# "Let me read /tmp/雪.txt:". Unlike the narrow ``let me now`` shape above it accepts plain ``let me`` and a
# dot inside a token (paths), but never crosses a quote mark, so a quoted example stays an answer.
# ``let me know`` is an offer, not an action.
_ANNOUNCED_ACTION_TAIL_RE = re.compile(
    r"(?:\blet me\s+(?:(?:now|first|just|also)\s+)?"
    r"(?:pull|read|inspect|check|open|run|search|fetch|fix|repair|edit|apply|restart|start|stop|pause|test|debug|compare|verify|look\s+at|look\s+up"
    r"|add|append|write|create|copy|install|configure|update|remove|delete|save|set\s+up)\b"
    r"|\bnext,? i(?:['\u2019]ll| will)\b)"
    r"(?:[^.!?\n\x22\u201c\u201d\u00ab\u00bb]|\.(?=[^\s.!?\x22\u201c\u201d\u00ab\u00bb])){0,100}[.:\u2026]?\s*$",
    re.IGNORECASE,
)
# An explicit request for time to do more tool work is unfinished even after a long report.
# Keep the historical short-ack limit for the broader first-person announcement patterns.
_PENDING_WORK_TAIL_RE = re.compile(
    r"(?:\bgive me\s+(?:a|one|another|a few|two)\s+(?:second|moment|minute)s?\s+to\s+"
    r"|\b(?:hang tight|bear with me)\s+while\s+i\s+)"
    r"(?:check|inspect|verify|search|research|read|open|fetch|look\s+up|look\s+at|compare"
    r"|run|test|fix|repair|build|install|configure|update|set\s+up)\b"
    r"(?:[^.!?\n\x22\u201c\u201d\u00ab\u00bb]|\.(?=[^\s.!?\x22\u201c\u201d\u00ab\u00bb])){0,160}"
    r"[.:\u2026]*\s*$",
    re.IGNORECASE,
)
# Requests that are not asking for action: a [discuss] turn, a question, or a pause/wait/approval stop.
# Anchored at the START of the request (request mode): a leading [discuss]/translate/advise/explain turn or a
# leading pause/wait/stop-for-approval. A wait/pause later in an execution request ("Run X, wait for it, then
# report") never vetoes.
# Explicit controls only: a leading [discuss]/[advice]/[translate] mode tag, or a WHOLE-message task pause
# ("Stop.", "Pause the repair.", "Stop here and wait.", "Pause and wait for my approval."). General
# questions/explain/describe/translate prose and stop/pause on arbitrary objects are NOT vetoes.
_NON_EXECUTION_REQUEST_RE = re.compile(r"^\s*\[(?:discuss|advice|advise|translat\w*)\b", re.IGNORECASE)
_PAUSE_REQUEST_RE = re.compile(
    r"^\s*(?:please[,\s]+)?(?:pause|stop|hold\s+off|wait)"
    r"(?:\s+(?:the|this|that|my|our)?\s*(?:task|work|repair))?"
    r"(?:\s+(?:here|now))*"
    r"(?:\s+and\s+(?:wait|await))?"
    r"(?:\s+(?:for|until)\s+(?:my|your|further|the)\s+(?:approval|instructions?|go-?ahead|confirmation|word))?"
    r"\s*[.!]*\s*$",
    re.IGNORECASE,
)
_QUOTE_PAIRS = (("\u201c", "\u201d"), ("\u00ab", "\u00bb"))
# Adding a clarification to the answer is a completed speech act, not a file mutation.
_ADDED_COMMENT_RE = re.compile(
    r"^let me\s+(?:(?:now|first|just|also)\s+)?add\s+"
    r"(?:that\b|(?:(?:a|an|one|another|this)\s+)?(?:clarification|note|point|caveat|comment)\s*:)",
    re.IGNORECASE,
)


def request_forbids_action(request: Any) -> bool:
    """Whether the latest user request explicitly asks for discussion/advice/translation/a pause (no tool action)."""
    r = request.strip() if isinstance(request, str) else ""
    if not r:
        return False
    return bool(
        _NON_EXECUTION_REQUEST_RE.search(r) or _PAUSE_REQUEST_RE.match(r) or _literal_output_request(r)
    )


# Explicit literal-output requests (finite coverage): a translation request with a supplied payload
# (colon / opening quote after "translate"), or a request to give/show/print/say/repeat/write an example
# sentence/phrase/quotation. Any "then"/"and" continuation (mixed request) is NOT a literal-output request.
_LITERAL_TRANSLATE_RE = re.compile(
    r"^\s*(?:please[,\s]+)?translate\b(?P<head>[^\n:\x22\u201c\u00ab]{0,80})[:\x22\u201c\u00ab]\s*\S",
    re.IGNORECASE,
)
_LITERAL_EXAMPLE_RE = re.compile(
    r"^\s*(?:please[,\s]+)?(?:give|show|print|say|repeat|write)(?:\s+me)?\s+(?:(?:an?|one|the)\s+)?example\s+"
    r"(?:of\s+)?(?:(?:an?|the)\s+)?(?:[\w-]+\s+){0,3}(?:sentence|phrase|quotation|quote)s?\b[^,;\n]*$",
    re.IGNORECASE,
)
_MIXED_REQUEST_RE = re.compile(r"\b(?:then|afterwards?|and)\b", re.IGNORECASE)


# Explicit operational continuation ("... and fix the failing test", "then run ..."): a mixed instruction, so not pure literal output.
# Deliberately narrow: a bare "and" (two languages, literal prose) is not execution.
_OPERATIONAL_CONTINUATION_RE = re.compile(
    r"\b(?:then|and|afterwards?|after that)\s+(?:please\s+)?"
    r"(?:fix|repair|run|test|inspect|check|read|write|edit|delete|deploy|restart|stop|start|apply)\b",
    re.IGNORECASE,
)


def _literal_output_request(r: str) -> bool:
    """Whether ``r`` explicitly asks only for literal text output (translation payload / example sentence)."""
    if _OPERATIONAL_CONTINUATION_RE.search(r):
        return False
    m = _LITERAL_TRANSLATE_RE.match(r)
    if m is not None:
        return not _MIXED_REQUEST_RE.search(m.group("head"))
    if _LITERAL_EXAMPLE_RE.match(r) is not None:
        return not re.search(r"\bthen\b", r, re.IGNORECASE)
    return False


def _inside_quote(text: str, pos: int) -> bool:
    """Whether ``pos`` sits inside an open straight-quote, backtick, or typographic quotation."""
    head = text[:pos]
    if head.count('"') % 2 or head.count("`") % 2:
        return True
    return any(head.count(o) > head.count(c) for o, c in _QUOTE_PAIRS) or _inside_single_quote(text, pos)


def _inside_single_quote(text: str, pos: int) -> bool:
    """Whether ``pos`` sits inside an open single-quoted span ('...' or \u2018...\u2019).

    Apostrophes between alphanumerics (contractions, possessives) are ordinary text. An opener is a
    straight quote not preceded by an alphanumeric and followed by a non-space, or a curly \u2018. A
    closer needs an open span, a non-space before it, and no alphanumeric after it.
    """
    is_open = False
    for i in range(min(pos, len(text))):
        ch = text[i]
        if ch not in "'\u2018\u2019":
            continue
        prev = text[i - 1] if i else ""
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if not is_open:
            if ch == "\u2018" or (ch == "'" and not prev.isalnum() and nxt and not nxt.isspace()):
                is_open = True
        elif ch != "\u2018" and prev and not prev.isspace() and not nxt.isalnum():
            is_open = False
    return is_open
_EXECUTION_QUESTION_RE = re.compile(r"^\s*(?:(?:please|hey)[, ]+)?(?:can|could|would|will)\s+you\b", re.IGNORECASE)
# An announced "wait" is a legitimate stop, never a stall.
# Only a passive declaration right after the first-person prefix ("Let me wait for you", "Let me pause here"),
# not a tool action whose object mentions waiting ("Let me pause the container", "why the job is waiting"),
# or an action explicitly conditioned on user approval ("Next, I'll deploy once you approve").
_PAUSE_ACTION_RE = re.compile(
    r"^(?:let me|next[,:]?\s+i(?:['\u2019]ll| will)|i(?:['\u2019]?ll| will)\s+now|now\s+i(?:['\u2019]ll| will))\s+(?:now\s+|just\s+)?"
    r"(?:(?:wait|await|hold off)\b|stand(?:ing)? by\b"
    r"|pause(?=\s*(?:$|[.:,;!\u2026]|(?:here|now|for|until|while)\b)))",
    re.IGNORECASE,
)
_APPROVAL_GATE_RE = re.compile(
    r"\b(?:once|after|when|if|as soon as)\s+you\s+(?:approve|confirm|authori[sz]e|say\s+go"
    r"|give\s+(?:me\s+)?the\s+go-?ahead)\b",
    re.IGNORECASE,
)
# Explicit user condition/offer anywhere in the assistant text (not just the matched tail span):
# approval clauses and "if you want" / "if you'd like" offers (straight or curly apostrophes).
# Deliberately narrow: generic conditionals such as "if build fails, inspect logs" are execution.
_USER_CONDITION_OR_OFFER_RE = re.compile(
    r"\b(?:once|after|when|if|as soon as|upon)\s+you\s+(?:approve|confirm|authori[sz]e|say\s+go"
    r"|give\s+(?:me\s+)?the\s+go-?ahead)\b"
    r"|\b(?:once|after|when|upon|with|pending)\s+your\s+(?:approval|confirmation|go-?ahead|authori[sz]ation)\b"
    r"|\bif\s+you(?:\s+would|['\u2019\u2018]d)?\s+(?:want|like|prefer|wish)\b",
    re.IGNORECASE,
)


def has_user_condition_or_offer(text: str) -> bool:
    """Whether ``text`` anywhere states an explicit user-approval condition or an offer ("if you want")."""
    return bool(_USER_CONDITION_OR_OFFER_RE.search(text or ""))


def unfulfilled_action_announcement(text: str, user_text: Any = None) -> bool:
    """Whether a reply ENDS on an announced next action that was not taken (stall-guard trigger).

    The narrow ``trailing_continue_intent`` shapes always count; the broader plain ``let me ...`` /
    ``next, I will ...`` tail counts only when ``user_text`` (the latest real user request) is an
    execution request. Announced waits/pauses and text echoed from the request (translation, quotes)
    never count. Explicit requests for time to do more work also count after longer reports.
    ``trailing_continue_intent`` itself is unchanged for existing callers.
    """
    t = (text or "").strip()
    if not t:
        return False
    request = user_text.strip() if isinstance(user_text, str) else ""
    tail = t[-320:]
    # The veto covers every visible-text candidate, narrow ``let me now`` included.
    if request_forbids_action(request):
        return False
    # Leading conditions/offers anywhere in the reply make it a user-gated stop, not a stall.
    if has_user_condition_or_offer(t):
        return False
    match = None
    if len(t) <= _TRAILING_CONTINUE_INTENT_MAX_CHARS:
        match = _TRAILING_CONTINUE_INTENT_RE.search(tail)
        if match is None and request:
            match = _ANNOUNCED_ACTION_TAIL_RE.search(tail)
    if match is None and request:
        match = _PENDING_WORK_TAIL_RE.search(tail)
    if match is None:
        return False
    # A quoted example ("...", `...`, typographic) is output, not an announcement.
    if _inside_quote(t, len(t) - len(tail) + match.start()):
        return False
    span = match.group(0)
    if _PAUSE_ACTION_RE.match(span.strip()) or _APPROVAL_GATE_RE.search(span) or _ADDED_COMMENT_RE.match(span.strip()):
        return False
    if request and span.strip(" .:\u2026\n").casefold() in request.casefold():
        return False
    return True


def latest_execution_request_text(messages: list[dict[str, Any]]) -> str:
    """Text of the latest REAL user row (a steer counts); synthetic nudge/scaffolding rows are skipped.

    Empty string when there is none, which keeps the broader announcement tail off.
    """
    from agent.context_compressor import ContextCompressor
    for msg in reversed(messages or ()):
        if not ContextCompressor._is_actionable_user_turn(msg):
            continue
        if ContextCompressor._is_synthetic_compression_user_turn(msg):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            text = content.strip()
        else:
            from agent.codex_responses_adapter import _summarize_user_message_for_log
            text = (_summarize_user_message_for_log(content) or "").strip()
        return text
    return ""
