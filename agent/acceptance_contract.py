"""Passive, session-bound acceptance contracts. No model calls or continuation loop."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

HEADER = "[Persistent completion contract]"
KINDS = {"documentation", "tests", "behavior", "runtime"}
CLAIM = re.compile(
    r"^\s*(?:yes[,!. ]+)?(?:done|finished|completed)\b|"
    r"\b(?:all|everything|(?:the |original |whole |parent |entire )?(?:task|goal|assignment))"
    r"[^.!?\n]{0,90}\b(?:complete|done|finished|addressed|green)\b",
    re.IGNORECASE,
)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()


def directory(agent=None) -> Path:
    db = getattr(agent, "_session_db", None)
    if db is not None and isinstance(getattr(db, "db_path", None), (str, Path)):
        home = Path(db.db_path).parent
    else:
        from hermes_constants import get_hermes_home
        home = get_hermes_home()
    return home / "completion-contracts"


def atomic_json(path: Path, value: Any, *, replace: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".contract-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(name, path)
        else:
            os.link(name, path)  # Atomic publish without replacing an existing parent.
    finally:
        if os.path.exists(name):
            os.unlink(name)


def validate(spec: dict) -> dict:
    if spec.get("schema") != "hermes.acceptance-contract.v1":
        raise ValueError("unsupported acceptance contract")
    tasks = spec.get("tasks")
    if not isinstance(tasks, list) or not 1 <= len(tasks) <= 32:
        raise ValueError("contract requires 1-32 tasks")
    if any(not isinstance(t, dict) for t in tasks):
        raise ValueError("tasks must be objects")
    ids = [t["id"] for t in tasks]
    if len(set(ids)) != len(ids) or spec.get("root") not in ids:
        raise ValueError("duplicate tasks or missing root")
    by_id = {t["id"]: t for t in tasks}
    for task in tasks:
        _validate_task(task)
        parent = task.get("parent")
        if task["id"] == spec["root"] and parent:
            raise ValueError("root cannot have a parent")
        if task["id"] != spec["root"] and parent not in by_id:
            raise ValueError("every child requires an existing parent")
        seen = {task["id"]}
        while parent:
            if parent in seen:
                raise ValueError("parent cycle")
            seen.add(parent)
            parent = by_id[parent].get("parent")
    size = sum(len(t["objective"]) + sum(len(c["description"]) for c in t["checks"]) for t in tasks)
    if size > 12000:
        raise ValueError("contract context exceeds its 12000-character budget")
    verifiers = [c["verifier"] for t in tasks for c in t["checks"] if c.get("verifier")]
    if len(verifiers) > 8 or sum(v.get("timeout_seconds", 10) for v in verifiers) > 30:
        raise ValueError("automatic verification exceeds its 8-check/30-second budget")
    return spec


def _validate_task(task: dict) -> None:
    if not isinstance(task.get("id"), str) or not task["id"] or len(task["id"]) > 80:
        raise ValueError("invalid task id")
    if not isinstance(task.get("objective"), str) or not 1 <= len(task["objective"]) <= 4000:
        raise ValueError("invalid objective")
    if task.get("type") not in {"documentation", "implementation", "delivery"}:
        raise ValueError("invalid task type")
    checks = task.get("checks")
    if not isinstance(checks, list) or not 1 <= len(checks) <= 16:
        raise ValueError("each task requires explicit checks")
    if any(not isinstance(c, dict) for c in checks):
        raise ValueError("checks must be objects")
    if len({c.get("id") for c in checks}) != len(checks):
        raise ValueError("duplicate acceptance check ids")
    kinds = {c.get("kind") for c in checks}
    if not kinds <= KINDS:
        raise ValueError("invalid evidence kind")
    if task["type"] in {"implementation", "delivery"} and "behavior" not in kinds:
        raise ValueError("implementation completion requires behavior acceptance")
    if task["type"] == "delivery" and "runtime" not in kinds:
        raise ValueError("delivery completion requires runtime acceptance")
    for check in checks:
        if (not isinstance(check.get("id"), str) or not 1 <= len(check["id"]) <= 80
                or not isinstance(check.get("description"), str)
                or not 1 <= len(check["description"]) <= 512):
            raise ValueError("bounded check id and description required")
        if check.get("verifier") is not None:
            from agent.acceptance_verification import validate_verifier
            validate_verifier(check["verifier"])


def register(session_id: str, spec: dict, *, root_dir: Path | None = None) -> str:
    """Install an operator-defined immutable contract; never replace an existing parent."""
    root_dir = root_dir or directory()
    validate(spec)
    raw = canonical(spec)
    sha = digest(raw)
    binding = root_dir / ("session-" + digest(session_id.encode()) + ".json")
    if binding.exists():
        if json.loads(binding.read_bytes()) != {"contract": sha}:
            raise ValueError("session already has a parent contract; explicit new session required")
        return sha
    path = root_dir / (sha + ".json")
    if path.exists() and path.read_bytes() != raw:
        raise ValueError("immutable contract changed")
    if not path.exists():
        atomic_json(path, spec)
    try:
        atomic_json(binding, {"contract": sha}, replace=False)
    except FileExistsError:
        if json.loads(binding.read_bytes()) != {"contract": sha}:
            raise ValueError("session already has a parent contract") from None
    return sha


def load(agent) -> tuple[dict, str] | None:
    """Resolve the owning profile and bounded session lineage, including restart/compaction."""
    root_dir = directory(agent)
    session = getattr(agent, "session_id", "") or ""
    db = getattr(agent, "_session_db", None)
    seen = set()
    while session and session not in seen and len(seen) < 32:
        seen.add(session)
        binding = root_dir / ("session-" + digest(session.encode()) + ".json")
        if binding.exists():
            sha = json.loads(binding.read_bytes())["contract"]
            if not re.fullmatch(r"[a-f0-9]{64}", sha):
                raise ValueError("invalid contract binding")
            raw = (root_dir / (sha + ".json")).read_bytes()
            if digest(raw) != sha:
                raise ValueError("acceptance contract changed")
            return validate(json.loads(raw)), sha
        row = db.get_session(session) if db is not None else None
        session = row.get("parent_session_id") if isinstance(row, dict) else None
    return None


def report(agent) -> dict | None:
    bound = load(agent)
    if bound is None:
        return None
    spec, sha = bound
    from agent.acceptance_verification import check_receipt
    results = {}
    for task in spec["tasks"]:
        checks = [
            {"id": check["id"], "description": check["description"],
             "passed": check_receipt(directory(agent), sha, task["id"], check)}
            for check in task["checks"]
        ]
        results[task["id"]] = {
            "objective": task["objective"], "checks": checks,
            "complete": all(c["passed"] for c in checks),
        }
    # Ancestors remain open whenever a descendant has outstanding acceptance.
    for _ in spec["tasks"]:
        for task in spec["tasks"]:
            if task.get("parent") and not results[task["id"]]["complete"]:
                results[task["parent"]]["complete"] = False
    return {"contract": sha, "root": spec["root"], "tasks": results}


def context(agent) -> str:
    try:
        result = report(agent)
    except (OSError, ValueError, KeyError, TypeError):
        return HEADER + "\nAcceptance state is unreadable. Completion cannot be verified."
    if result is None:
        return ""
    lines = [HEADER, "The parent objective remains authoritative across subtasks and compression."]
    for task_id, task in result["tasks"].items():
        status = "verified" if task["complete"] else "open"
        lines.append(f"- {task_id} [{status}]: {task['objective']}")
        lines.extend(f"  - {'PASS' if c['passed'] else 'PENDING'} {c['id']}: {c['description']}"
                     for c in task["checks"])
    lines.append("A subtask can finish while the parent stays open. Report the scope accurately.")
    return "\n".join(lines)


def fold_snapshots(agent, messages, fold_todos) -> None:
    """Compression restores objective/checks from durable state, never from model summaries."""
    fold_todos(agent, messages)
    text = context(agent)
    if not text:
        return
    # Only edit the newly constructed compressed history, never the cached prefix.
    for message in list(messages):
        content = message.get("content")
        start = message.pop("_acceptance_context_start", None)
        if isinstance(content, str) and type(start) is int:
            message["content"] = content[:start]
        elif (message.get("_todo_snapshot_synthetic") and isinstance(content, str)
              and content.startswith(HEADER)):
            messages.remove(message)
    if messages and messages[-1].get("role") == "user":
        content = messages[-1].get("content")
        if isinstance(content, str):
            messages[-1]["_acceptance_context_start"] = len(content)
            messages[-1]["content"] = content + "\n\n" + text
            return
    messages.append({"role": "user", "content": text, "_todo_snapshot_synthetic": True})


def guarded(agent) -> bool:
    try:
        return load(agent) is not None
    except (OSError, ValueError, KeyError, TypeError):
        return True


def _positive_matches(pattern, text):
    from agent.completion_announcements import _inside_quote
    return [m for m in pattern.finditer(text or "")
            if not _inside_quote(text, m.start())
            and not re.search(r"\b(?:not|never|incomplete)\b", m.group(0), re.IGNORECASE)]


def _refresh(agent, spec, sha, selected):
    import subprocess
    from agent.acceptance_verification import verify
    if getattr(agent, "_interrupt_requested", False):
        return
    turn = getattr(agent, "_current_turn_id", None)
    scope = (getattr(agent, "session_id", ""), turn, sha)
    if getattr(agent, "_acceptance_refresh_scope", None) != scope:
        agent._acceptance_refresh_scope = scope
        agent._acceptance_refreshed_checks = set()
    for task in spec["tasks"]:
        if selected is not None and task["id"] not in selected:
            continue
        for check in task["checks"]:
            key = (task["id"], check["id"])
            seen = turn and key in agent._acceptance_refreshed_checks
            if check.get("verifier") and not seen:
                agent._acceptance_refreshed_checks.add(key)
                try:
                    verify(directory(agent), spec, sha, task["id"], check["id"])
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    # The verifier invalidates any old pass before it runs.
                    continue


def enforce_final(agent, text: str, message: dict) -> tuple[str, bool]:
    """Refresh reviewed read-only acceptance before delivering a completion claim."""
    broad = _positive_matches(CLAIM, text)
    if not broad and re.match(r"^\s*(?:yes|yep|correct|absolutely)\b", text or "", re.IGNORECASE):
        from agent.completion_announcements import latest_execution_request_text
        request = latest_execution_request_text(getattr(agent, "_session_messages", []))
        qualified = re.search(r"\b(?:incomplete|not complete|not done|remains open|still open)\b",
                              text, re.IGNORECASE)
        if not qualified and _positive_matches(CLAIM, request):
            broad = [True]
    try:
        bound = load(agent)
    except (OSError, ValueError, KeyError, TypeError):
        bound = None
        if not broad:
            return text, False
        result = {"root": "unreadable", "tasks": {}}
    else:
        if bound is None:
            return text, False
        spec, sha = bound
        scoped = [t["id"] for t in spec["tasks"] if _positive_matches(
            re.compile(r"\b" + re.escape(t["id"]) +
                       r"\b[^.!?\n]{0,60}\b(?:complete|done|finished|addressed)\b",
                       re.IGNORECASE), text)]
        if not broad and not scoped:
            return text, False
        _refresh(agent, spec, sha, None if broad else scoped)
        result = report(agent)
        targets = [result["root"]] if broad else scoped
        if all(result["tasks"][t]["complete"] for t in targets):
            return text, False
    verified = [k for k, t in result["tasks"].items() if t["complete"]]
    pending = [c["description"] for t in result["tasks"].values() for c in t["checks"] if not c["passed"]]
    safe = "The parent objective remains open."
    if verified:
        safe += " Verified subtasks: " + ", ".join(verified) + "."
    safe += " Outstanding acceptance: " + ("; ".join(pending[:12]) or "acceptance state could not be verified") + "."
    message.pop("api_content", None)
    message["content"] = safe
    message["acceptance_contract"] = {"root": result["root"], "complete": False}
    return safe, True


def bind_verifier(agent, expected: str, task_id: str, check_id: str, verifier: dict) -> str:
    """Explicit operator amendment changes one verifier, preserving every objective and check."""
    from agent.acceptance_verification import validate_verifier
    validate_verifier(verifier)
    root_dir = directory(agent)
    binding = root_dir / ("session-" + digest(agent.session_id.encode()) + ".json")
    lock = binding.with_suffix(".lock")
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        bound = load(agent)
        if bound is None or bound[1] != expected:
            raise ValueError("parent contract changed; inspect current state first")
        spec, _ = bound
        task = next(t for t in spec["tasks"] if t["id"] == task_id)
        check = next(c for c in task["checks"] if c["id"] == check_id)
        check["verifier"] = verifier
        spec["previous_contract"] = expected
        validate(spec)
        sha = digest(canonical(spec))
        atomic_json(root_dir / (sha + ".json"), spec)
        atomic_json(binding, {"contract": sha})
        return sha
    finally:
        os.close(fd)
        lock.unlink()


def reconcile_output(agent, text, message, promoted, transformed):
    """Keep the checked response and durable row in agreement after an output transform."""
    text, blocked = enforce_final(agent, text, message)
    if transformed and not blocked:
        message["api_content" if promoted else "content"] = text
    return text
