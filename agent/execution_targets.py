"""Profile-configured execution targets, retained independently of compressed prose.

This constrains supported host requests; it never grants SSH access or runs a command.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import threading

from hermes_yaml import YAMLError

from agent.acceptance_contract import atomic_json

_ACTION = re.compile(r"\b(?:check|inspect|read|search|find|run|test|fix|edit|verify|commit|continue|resume|use|switch|work|compare)\b", re.IGNORECASE)
_FOLLOWUP = re.compile(r"^\s*(?:yes(?: please)?|please do|do it|continue|resume|go ahead)[.!\s]*$", re.IGNORECASE)
_RESOURCE = re.compile(r"\b(?:files?|repository|repo|code|commit|diff|worktree|project|directory|path)\b", re.IGNORECASE)
_CLEAR = re.compile(r"\b(?:change gears|switch topics|new topic|never mind|nevermind)\b", re.IGNORECASE)


def home(agent) -> Path:
    db = getattr(agent, "_session_db", None)
    if isinstance(getattr(db, "db_path", None), (str, Path)):
        return Path(db.db_path).parent
    from hermes_constants import get_hermes_home
    return get_hermes_home()


def _path(agent, session=None) -> Path:
    sid = session or getattr(agent, "session_id", "") or ""
    key = hashlib.sha256(sid.encode()).hexdigest()
    return home(agent) / "execution-targets" / (key + ".json")


def _config(agent) -> tuple[list, str]:
    from utils import load_yaml_file_readonly
    path = home(agent) / "config.yaml"
    raw = load_yaml_file_readonly(path) if path.exists() else {}
    if not isinstance(raw, dict):
        raise ValueError("execution target configuration must be a mapping")
    settings = raw.get("agent") or {}
    if not isinstance(settings, dict):
        raise ValueError("agent configuration must be a mapping")
    entries = settings.get("execution_targets", [])
    if not isinstance(entries, list) or len(entries) > 8:
        raise ValueError("execution_targets must be a list of at most eight targets")
    for entry in entries:
        _validate(entry)
    ids = [entry["id"] for entry in entries]
    if len(set(ids)) != len(ids):
        raise ValueError("execution target ids must be unique")
    aliases = [alias.casefold() for entry in entries for alias in entry["aliases"]]
    if len(set(aliases)) != len(aliases):
        raise ValueError("execution target aliases must be unique")
    digest = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    return entries, digest


def _validate(entry) -> None:
    if not isinstance(entry, dict) or not re.fullmatch(r"[a-z0-9_-]{1,40}", entry.get("id", "")):
        raise ValueError("invalid execution target id")
    aliases = entry.get("aliases")
    if not isinstance(aliases, list) or not 1 <= len(aliases) <= 8 or any(
        not isinstance(a, str) or not 1 <= len(a) <= 80 for a in aliases
    ):
        raise ValueError("execution targets need bounded aliases")
    if not entry.get("local") and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", entry.get("destination", "")):
        raise ValueError("execution target needs an SSH destination alias")
    projects = entry.get("projects", {})
    if not isinstance(projects, dict) or len(projects) > 16 or any(
        not isinstance(k, str) or not 1 <= len(k) <= 80 or not isinstance(v, str)
        or not v.startswith("/") or len(v) > 1024 or "\n" in v or "\0" in v
        for k, v in projects.items()
    ):
        raise ValueError("execution projects need bounded absolute paths")


def prose(request) -> str:
    """Drop the gateway's quoted reply and fenced/blockquote data before detecting a request."""
    if not isinstance(request, str):
        return ""
    if request.startswith("[Replying to") and '"]\n\n' in request:
        request = request.rsplit('"]\n\n', 1)[1]
    request = re.sub(r"```.*?```", "", request, flags=re.DOTALL)
    return "\n".join(line for line in request.splitlines() if not line.lstrip().startswith(">"))


def _mentions(text, aliases) -> bool:
    from agent.completion_announcements import _inside_quote
    for alias in aliases:
        for match in re.finditer(r"(?<![\w-])" + re.escape(alias) + r"(?![\w-])", text, re.IGNORECASE):
            if not _inside_quote(text, match.start()) or text[max(0, match.start()-1):match.end()+1] == f"`{match.group()}`":
                return True
    return False


def _saved(agent, digest):
    session = getattr(agent, "session_id", "") or ""
    db, seen = getattr(agent, "_session_db", None), set()
    while session and session not in seen and len(seen) < 32:
        seen.add(session)
        path = _path(agent, session)
        if path.exists():
            value = json.loads(path.read_bytes())
            if not isinstance(value, dict):
                raise ValueError("execution target record must be a mapping")
            if value.get("config") != digest:
                raise ValueError("execution target configuration changed; restate the target")
            return value
        row = db.get_session(session) if db is not None else None
        session = row.get("parent_session_id") if isinstance(row, dict) else None
    return None


def begin(agent, request) -> None:
    """Resolve a target at a turn/steer boundary, excluding late worker results."""
    if not hasattr(agent, "_execution_target_lock"):
        agent._execution_target_lock = threading.Lock()
    with agent._execution_target_lock:
        _begin(agent, request)


def _begin(agent, request) -> None:
    agent._execution_target = None
    agent._execution_target_needed = agent._execution_target_verified = False
    agent._execution_target_blocked = agent._execution_target_failed = False
    agent._execution_target_unverified = False
    agent._execution_target_preexisting_failure = False
    agent._execution_target_refusals = 0
    agent._execution_target_error = ""
    agent._execution_target_epoch = object()
    agent._acceptance_stream_guard = None
    try:
        entries, digest = _config(agent)
        if not entries:
            return
        from agent.completion_announcements import request_forbids_action
        text = prose(request)
        active_request = not request_forbids_action(text) and bool(_ACTION.search(text))
        matches = [e for e in entries if active_request and _mentions(text, e["aliases"])]
        if len(matches) > 1 or (_CLEAR.search(text) and not matches):
            atomic_json(_path(agent), {"config": digest, "target": None})
            return  # Multi-host tasks retain their normal tool behavior; no single-host inference.
        saved = _saved(agent, digest) if not matches else None
        value = saved
        if matches:
            entry = matches[0]
            projects = [k for k in entry.get("projects", {}) if _mentions(text, [k])]
            value = {"config": digest, "target": entry["id"],
                     "project": projects[0] if len(projects) == 1 else None}
            atomic_json(_path(agent), value)
        if not value or value.get("target") is None:
            return
        entry = next(e for e in entries if e["id"] == value["target"])
        if entry.get("local"):
            return
        agent._execution_target = {**entry, "workspace": entry.get("projects", {}).get(value.get("project"))}
        agent._execution_target_needed = bool(matches) or bool(_FOLLOWUP.fullmatch(text)) or bool(active_request and _RESOURCE.search(text))
    except (OSError, ValueError, KeyError, TypeError, StopIteration, YAMLError):
        agent._execution_target_error = "The configured execution target could not be restored. Restate the host after checking its configuration."
        agent._execution_target_needed = True


def context(agent) -> str:
    error = getattr(agent, "_execution_target_error", "")
    if error:
        return "[Execution target]\n" + error
    target = getattr(agent, "_execution_target", None)
    if not target:
        return ""
    workspace = target.get("workspace")
    return (f"[Execution target]\nThis task targets {target['id']} through SSH alias {target['destination']}. "
            + (f"Its project workspace is {workspace}. " if workspace else "")
            + "Local filesystem results from the Hermes machine cannot verify this target. "
            "Use one SSH command to the configured alias, with the project directory in the remote command. "
            "A failed or rejected tool call leaves the requested verification unfinished.")


def guarded(agent) -> bool:
    return bool(getattr(agent, "_execution_target_needed", False) or getattr(agent, "_execution_target_blocked", False))


def enforce(agent, text, message):
    if getattr(agent, "_interrupt_requested", False) or getattr(agent, "_execution_target_preexisting_failure", False):
        return text, False
    if not guarded(agent) or getattr(agent, "_execution_target_verified", False):
        return text, False
    target = getattr(agent, "_execution_target", None)
    label = target["id"] if target else "the requested host"
    safe = f"I have not verified {label} in this turn. "
    safe += ("The target command failed. " if getattr(agent, "_execution_target_failed", False)
             else "No successful command against the requested target was recorded. ")
    safe += "The requested repository result remains unverified."
    agent._execution_target_unverified = True
    message.pop("api_content", None)
    message["content"] = safe
    return safe, True


def finish_status(agent, text, failed, reason, interrupted):
    agent._execution_target_preexisting_failure = bool(failed)
    if interrupted or failed:
        return text, failed, reason
    text, blocked = enforce(agent, text, {})
    return (text, True, "execution_target_unverified") if blocked else (text, failed, reason)
