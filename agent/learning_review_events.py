"""Body-free, profile-scoped evidence for background learning reviews.

Events describe execution and successful tool write results. They do not score
the quality of a lesson or prove that another conversation used it.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
EVENTS = frozenset({"queued", "started", "cancel_requested", "completed", "incomplete", "cancelled", "failed", "skipped", "unknown"})
TERMINAL_EVENTS = EVENTS - {"queued", "started", "cancel_requested"}
_MUTATIONS = {"memory": ("add", "replace", "remove"),
              "skill_manage": ("create", "edit", "patch", "write_file", "remove_file", "delete")}


def event_path(home: Path) -> Path:
    return home / "logs" / "learning-reviews.jsonl"


def record_decision(agent: Any, event: str, reason: str) -> None:
    record_event(agent, uuid.uuid4().hex, event, reason=reason)


def _append(path: Path, row: dict) -> None:
    from tools.skill_usage import skill_file_lock

    path.parent.mkdir(parents=True, exist_ok=True)
    with skill_file_lock(path.with_suffix(".lock")):
        if path.exists() and path.stat().st_size >= 4 * 1024 * 1024:
            os.replace(path, path.with_suffix(".jsonl.1"))
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "ab") as stream:
            stream.write((json.dumps(row, ensure_ascii=True) + "\n").encode())


def record_event(agent: Any, review_id: str, event: str, *, reason: str | None = None,
                 error_type: str | None = None, writes: dict | None = None) -> None:
    """Best effort: a telemetry failure never changes foreground or review behavior."""
    try:
        from agent.system_prompt import _agent_home
        from hermes_constants import get_hermes_home

        if event not in EVENTS:
            raise ValueError("unknown review event")
        row = {"version": 1, "time": datetime.now(UTC).isoformat(),
               "session_id": str(getattr(agent, "session_id", "") or ""),
               "review_id": review_id, "event": event}
        if reason:
            row["reason"] = reason
        if error_type:
            row["error_type"] = error_type
        if writes is not None:
            row["writes"] = {key: int(writes.get(key, 0)) for key in ("memory", "skill", "staged")}
        _append(event_path(_agent_home(agent) or get_hermes_home()), row)
    except Exception as exc:  # health: allow BLE001 -- telemetry must not block execution; traceback text can expose private tool content
        # Exception messages can contain prompts, credentials, or filesystem content.
        logger.warning("Could not record learning review event (%s)", type(exc).__name__)


def review_outcome(result: Any, cancelled: bool) -> str:
    if cancelled or (isinstance(result, dict) and result.get("interrupted")):
        return "cancelled"
    if not isinstance(result, dict):
        return "unknown"
    if result.get("error"):
        return "failed"
    if result.get("partial") or result.get("completed") is False:
        return "incomplete"
    return "completed" if result.get("completed") is True else "unknown"


def _decode_dict(value: Any) -> dict:
    try:
        result = json.loads(value) if isinstance(value, str) else value
    except (ValueError, TypeError):
        return {}
    return result if isinstance(result, dict) else {}


def _review_calls(messages: list[dict]) -> dict:
    calls = {}
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        tool_calls = msg.get("tool_calls")
        for call in tool_calls if isinstance(tool_calls, list) else []:
            if not isinstance(call, dict) or not isinstance(call.get("id"), str):
                continue
            fn = call.get("function") or {}
            if isinstance(fn, dict) and isinstance(fn.get("name"), str) and fn["name"] in _MUTATIONS:
                calls[call.get("id")] = (fn["name"], _decode_dict(fn.get("arguments")))
    return calls


def _write_result(tool: str, args: dict, data: dict) -> tuple[str, int]:
    if data.get("success") is not True or data.get("operations_applied") is False:
        return "staged", 0
    if data.get("staged"):
        return "staged", 1
    key = "memory" if tool == "memory" else "skill"
    if data.get("message") == "Entry already exists (no duplicate added).":
        return key, 0
    if tool == "skill_manage" and isinstance(data.get("results"), list):
        applied = sum(isinstance(r, dict) and r.get("success") is True
                      and r.get("action") in _MUTATIONS[tool] for r in data["results"])
        return key, applied if data.get("operations_applied") else 0
    if tool == "memory" and isinstance(args.get("operations"), list):
        return key, len(args["operations"])
    return key, int(args.get("action") in _MUTATIONS[tool])


def write_counts(messages: list[dict], prior: list[dict]) -> dict[str, int]:
    """Count only attributed, fresh mutation results. Staging and rollback stay separate."""
    old_ids = {m["tool_call_id"] for m in prior if isinstance(m, dict) and m.get("role") == "tool"
               and isinstance(m.get("tool_call_id"), str)}
    calls = _review_calls(messages)
    counts = {"memory": 0, "skill": 0, "staged": 0}
    seen = set(old_ids)
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "tool":
            continue
        call_id = msg.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id or call_id in seen or call_id not in calls:
            continue
        seen.add(call_id)
        tool, args = calls[call_id]
        key, count = _write_result(tool, args, _decode_dict(msg.get("content")))
        counts[key] += count
    return counts
