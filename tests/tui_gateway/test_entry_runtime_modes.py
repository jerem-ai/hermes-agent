"""Direct stdio gateway runtime-mode contract tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from hermes_cli.broker_mode import (
    _BROKER_MODE_MARKER,
    _QUALIFICATION_MODE_MARKER,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _isolated_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.pop(_BROKER_MODE_MARKER, None)
    env.pop(_QUALIFICATION_MODE_MARKER, None)
    env["HOME"] = str(tmp_path / "home")
    env["HERMES_HOME"] = str(tmp_path / "hermes-home")
    env["PYTHONPATH"] = str(REPO_ROOT)
    return env


def test_python_module_entry_accepts_public_runtime_flags(tmp_path):
    """The production ``python -m`` entry reaches ready state with both flags."""

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tui_gateway.entry",
            "--qualification-mode",
            "--broker-mode",
        ],
        cwd=REPO_ROOT,
        env=_isolated_env(tmp_path),
        input="",
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    events = [
        json.loads(line)
        for line in completed.stdout.splitlines()
        if line.startswith("{")
    ]
    assert any(
        event.get("params", {}).get("type") == "gateway.ready"
        for event in events
    ), (completed.stdout, completed.stderr)


def test_direct_entry_public_flags_enable_tool_free_runtime_before_server_import(
    tmp_path,
):
    """The supported direct entry rejects a hostile tool call before dispatch."""

    script = r'''
import json
import sys
from types import SimpleNamespace

sys.argv = [
    "tui_gateway.entry",
    "--qualification-mode",
    "--broker-mode",
    "--caller-owned-argument",
]

import tui_gateway.entry
import model_tools
from hermes_cli.broker_mode import broker_mode_requested, qualification_mode_requested
from hermes_cli.runtime_identity import effective_runtime_identity

identity = effective_runtime_identity(SimpleNamespace(
    provider="custom",
    model="hostile/model",
    base_url="http://127.0.0.1:11434/v1?token=must-not-leak",
))
result = {
    "argv": sys.argv,
    "broker_mode": broker_mode_requested(),
    "qualification_mode": qualification_mode_requested(),
    "tools": model_tools.get_tool_definitions(quiet_mode=True),
    "tool_result": model_tools.handle_function_call(
        "terminal",
        {"command": "touch escaped"},
    ),
    "runtime_identity": identity,
}
print(
    "DIRECT_ENTRY_RESULT=" + json.dumps(result, sort_keys=True),
    file=sys.__stdout__,
    flush=True,
)
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=_isolated_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    result_lines = [
        line
        for line in completed.stdout.splitlines()
        if line.startswith("DIRECT_ENTRY_RESULT=")
    ]
    assert result_lines, (completed.stdout, completed.stderr)
    result_line = result_lines[-1]
    result = json.loads(result_line.split("=", 1)[1])

    assert result["argv"] == ["tui_gateway.entry", "--caller-owned-argument"]
    assert result["broker_mode"] is True
    assert result["qualification_mode"] is True
    assert result["tools"] == []
    assert "disabled in qualification mode" in result["tool_result"]
    assert result["runtime_identity"]["broker"] == {
        "contract_version": 1,
        "mode": True,
        "available": False,
    }
    assert result["runtime_identity"]["qualification"] == {
        "contract_version": 1,
        "mode": True,
        "tool_free": True,
        "tool_surface": "empty",
        "tool_execution": "rejected",
        "sampling": {
            "temperature": 0,
            "seed": 20260830,
            "max_output_tokens": 256,
        },
    }
    serialized = json.dumps(result["runtime_identity"], sort_keys=True)
    assert "must-not-leak" not in serialized
    assert "127.0.0.1" not in serialized
