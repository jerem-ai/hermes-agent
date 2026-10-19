"""Explicit verification of pinned acceptance programs and their subject-bound receipts."""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import time

from agent.acceptance_contract import atomic_json, canonical, digest


def validate_verifier(verifier: dict) -> None:
    argv = verifier.get("argv")
    if not isinstance(argv, list) or not argv or len(argv) > 32:
        raise ValueError("verifier needs an argv array")
    if any(not isinstance(v, str) or not v or "\0" in v for v in argv):
        raise ValueError("invalid verifier argv")
    pins = verifier.get("files")
    if not isinstance(pins, dict) or not pins or len(pins) > 32:
        raise ValueError("pin verifier source and acceptance subjects")
    if any(not Path(p).is_absolute() or len(sha) != 64 for p, sha in pins.items()):
        raise ValueError("verifier files require absolute paths and SHA-256")
    executable = Path(argv[0])
    if not executable.is_absolute():
        raise ValueError("verifier executable must be absolute")
    script_index = 1 if executable.name.startswith(("python", "node", "ruby")) else 0
    if len(argv) <= script_index or argv[script_index] not in pins:
        raise ValueError("verifier program must be source-pinned; inline shell/code is refused")
    if not Path(argv[script_index]).is_absolute():
        raise ValueError("verifier program must be absolute")
    timeout = verifier.get("timeout_seconds", 10)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 30:
        raise ValueError("verifier timeout must be 0-30 seconds")
    age = verifier.get("maximum_age_seconds", 3600)
    if type(age) not in (int, float) or not math.isfinite(age) or not 0 < age <= 604800:
        raise ValueError("receipt lifetime must be positive and bounded")


def pins_match(verifier: dict) -> bool:
    for name, sha in verifier["files"].items():
        path = Path(name)
        if path.is_symlink() or not path.is_file() or digest(path.read_bytes()) != sha:
            return False
    return True


def receipt_path(root: Path, sha: str, task: str, check: str) -> Path:
    return root / "receipts" / (digest(canonical([sha, task, check])) + ".json")


def verify(root: Path, spec: dict, sha: str, task_id: str, check_id: str) -> dict:
    """Only an explicit CLI invocation runs an operator-defined verification command."""
    task = next(t for t in spec["tasks"] if t["id"] == task_id)
    check = next(c for c in task["checks"] if c["id"] == check_id)
    verifier = check.get("verifier")
    if verifier is None:
        raise ValueError("acceptance has no independent verifier yet")
    validate_verifier(verifier)
    if not pins_match(verifier):
        raise ValueError("verifier source or acceptance subject changed")
    started = time.time()
    path = receipt_path(root, sha, task_id, check_id)
    atomic_json(path, {"passed": False, "observed_at": started})
    result = subprocess.run(
        verifier["argv"], cwd=verifier.get("cwd"), stdin=subprocess.DEVNULL,
        capture_output=True, timeout=verifier.get("timeout_seconds", 10), shell=False, check=False,
    )
    try:
        output = json.loads(result.stdout) if len(result.stdout) <= 65536 else {}
    except (ValueError, UnicodeError):
        output = {}
    passed = result.returncode == 0 and isinstance(output, dict) and output.get("passed") is True
    receipt = {
        "schema": "hermes.acceptance-receipt.v1", "contract": sha, "task": task_id,
        "check": check_id, "kind": check["kind"], "exit_code": result.returncode,
        "passed": passed and pins_match(verifier), "observed_at": started,
        "verifier_sha256": digest(canonical(verifier)),
    }
    atomic_json(receipt_path(root, sha, task_id, check_id), receipt)
    return receipt


def check_receipt(root: Path, sha: str, task: str, check: dict) -> bool:
    verifier = check.get("verifier")
    if verifier is None:
        return False
    path = receipt_path(root, sha, task, check["id"])
    try:
        row = json.loads(path.read_bytes())
        elapsed = time.time() - row["observed_at"]
        return (
            row.get("schema") == "hermes.acceptance-receipt.v1"
            and row.get("contract") == sha and row.get("task") == task
            and row.get("check") == check["id"] and row.get("kind") == check["kind"]
            and row.get("passed") is True and row.get("exit_code") == 0
            and row.get("verifier_sha256") == digest(canonical(verifier))
            and -1 <= elapsed <= verifier.get("maximum_age_seconds", 3600)
            and pins_match(verifier)
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
