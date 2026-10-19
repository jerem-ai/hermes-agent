"""Constrain filesystem calls after argument middleware, using a turn's declared host."""
from __future__ import annotations

import json
import shlex
from pathlib import PurePosixPath

_FILESYSTEM = frozenset({"read_file", "write_file", "patch", "search_files", "execute_code", "delegate_task"})
_IDENTITY = frozenset({"hostname", "pwd", "whoami"})
_SSH_FLAGS = frozenset({"-T", "-t", "-tt", "-n"})
_SSH_OPTIONS = ("BatchMode=", "ConnectTimeout=", "ServerAliveInterval=", "ServerAliveCountMax=")


def _ssh_command(command, destination):
    """Accept a single SSH invocation with transport-only options and one remote shell string."""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except (TypeError, ValueError):
        return None
    if not tokens or tokens.pop(0) not in {"ssh", "/usr/bin/ssh"}:
        return None
    while tokens and tokens[0].startswith("-"):
        flag = tokens.pop(0)
        if flag == "--":
            break
        if flag in _SSH_FLAGS:
            continue
        if flag == "-o" and tokens:
            option = tokens.pop(0)
        elif flag.startswith("-o"):
            option = flag[2:]
        else:
            return None
        if not option.startswith(_SSH_OPTIONS):
            return None
    if len(tokens) != 2 or tokens[0] != destination:
        return None
    return tokens[1]


def _workspace_command(command, workspace):
    if command.strip() in _IDENTITY:
        return False
    if not workspace:
        return True
    try:
        words = shlex.split(command)
    except ValueError:
        return False
    return (words[:3] == ["git", "-C", workspace]
            or words[:3] == ["cd", workspace, "&&"]
            or words[:4] == ["cd", "--", workspace, "&&"])


def allowed(agent, name, args) -> bool:
    target = getattr(agent, "_execution_target", None)
    if not target:
        return False
    if name == "terminal" and args.get("command", "").strip() in _IDENTITY:
        return False
    if _native_target(target, name, args):
        return True
    if name != "terminal":
        return False
    remote = _ssh_command(args.get("command", ""), target["destination"])
    return remote is not None and _workspace_command(remote, target.get("workspace"))


def _native_target(target, name, args):
    from tools.terminal_scope import terminal_env
    if terminal_env("TERMINAL_ENV", "local") != "ssh" or terminal_env("TERMINAL_SSH_HOST") != target["destination"]:
        return False
    if name not in {"terminal", "read_file", "write_file", "patch", "search_files"}:
        return False
    workspace = target.get("workspace")
    path = args.get("path") or args.get("workdir") or terminal_env("TERMINAL_CWD")
    if not workspace:
        return True
    return (isinstance(path, str) and path.startswith("/") and ".." not in PurePosixPath(path).parts
            and PurePosixPath(path).is_relative_to(PurePosixPath(workspace)))


def block(agent, name, args):
    if name not in _FILESYSTEM and name != "terminal":
        return None
    epoch = getattr(agent, "_execution_target_epoch", None)
    target = getattr(agent, "_execution_target", None)
    error = getattr(agent, "_execution_target_error", "")
    if not target and not error:
        return None
    if target and name == "terminal" and args.get("command", "").strip() in _IDENTITY:
        return None
    if target and name == "terminal" and _ssh_command(args.get("command", ""), target["destination"]) in _IDENTITY:
        return None
    if not error and allowed(agent, name, args):
        return None
    from hermes_platform.host.facts import hostname
    message = error or (f"Execution target mismatch: these tools start on {hostname()}, while this task targets "
                       f"{target['id']} via {target['destination']}. "
                       + (f"Use the remote workspace {target['workspace']}. " if target.get("workspace") else "")
                       + "Use terminal with one quoted SSH command to that alias. This call was not executed.")
    _count_refusal(agent, name, message, epoch)
    return message


def _count_refusal(agent, name, message, epoch):
    from agent.tool_guardrails import ToolGuardrailDecision
    with agent._execution_target_lock:
        if epoch is not agent._execution_target_epoch:
            return
        agent._execution_target_blocked = True
        agent._execution_target_refusals += 1
        if agent._execution_target_refusals >= 3:
            agent._tool_guardrail_halt_decision = ToolGuardrailDecision(
                action="halt", code="execution_target_mismatch", message=message,
                tool_name=name, count=agent._execution_target_refusals)


def record(agent, name, args, result, *, epoch=None):
    lock = getattr(agent, "_execution_target_lock", None)
    if lock is None:
        return result
    with lock:
        return _record(agent, name, args, result, epoch=epoch)


def _record(agent, name, args, result, *, epoch):
    if epoch is not None and epoch is not getattr(agent, "_execution_target_epoch", None):
        return result
    if not allowed(agent, name, args):
        return result
    try:
        value = json.loads(result) if isinstance(result, str) else result
        success = isinstance(value, dict) and not value.get("error") and (
            value.get("exit_code") == 0 if name == "terminal"
            else "content" in value or value.get("success") is True or value.get("status") == "success")
    except (TypeError, ValueError):
        success = False
    if success:
        agent._execution_target_verified = True
        agent._execution_target_refusals = 0
    else:
        agent._execution_target_failed = True
    return result
