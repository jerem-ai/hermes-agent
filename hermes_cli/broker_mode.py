"""Process-boundary helpers for the supported broker launch mode."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence


_BROKER_MODE_MARKER = "_HERMES_INTERNAL_BROKER_MODE"
_QUALIFICATION_MODE_MARKER = "_HERMES_INTERNAL_QUALIFICATION_MODE"
_BROKER_ROUTE_ID = "HERMES_BROKER_ROUTE_ID"
BROKER_CHANNEL_FD = 3
BROKER_PROTOCOL_VERSION = 1
QUALIFICATION_TEMPERATURE = 0
QUALIFICATION_SEED = 20260830
QUALIFICATION_MAX_OUTPUT_TOKENS = 256


def broker_mode_requested(
    argv: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return whether this process belongs to a ``--broker-mode`` launch.

    The private marker carries the CLI decision through the Python -> Node ->
    Python TUI process chain. It is an internal transport detail, not a user
    configuration surface and never indicates that a broker is available.
    """

    args = sys.argv[1:] if argv is None else argv
    env = os.environ if environ is None else environ
    return "--broker-mode" in args or env.get(_BROKER_MODE_MARKER) == "1"


def qualification_mode_requested(
    argv: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Return whether this process belongs to a tool-free qualification run."""

    args = sys.argv[1:] if argv is None else argv
    env = os.environ if environ is None else environ
    return (
        "--qualification-mode" in args
        or env.get(_QUALIFICATION_MODE_MARKER) == "1"
    )


def broker_route_id(environ: Mapping[str, str] | None = None) -> str:
    """Return the opaque route bound to this broker-mode process."""

    env = os.environ if environ is None else environ
    value = str(env.get(_BROKER_ROUTE_ID, "") or "").strip()
    if not value or len(value) > 200 or any(ord(char) < 32 for char in value):
        raise RuntimeError("Broker mode requires a valid opaque route ID")
    return value


def apply_broker_mode_marker(env: dict[str, str], *, enabled: bool) -> None:
    """Project broker mode into a child environment without stale inheritance."""

    if enabled:
        env[_BROKER_MODE_MARKER] = "1"
    else:
        env.pop(_BROKER_MODE_MARKER, None)


def apply_qualification_mode_marker(env: dict[str, str], *, enabled: bool) -> None:
    """Project qualification mode into a child environment."""

    if enabled:
        env[_QUALIFICATION_MODE_MARKER] = "1"
    else:
        env.pop(_QUALIFICATION_MODE_MARKER, None)


def consume_runtime_mode_flags(
    argv: Sequence[str],
    env: dict[str, str],
) -> list[str]:
    """Translate supported public entry flags into process-local markers.

    The stdio gateway is also launched directly with ``python -m
    tui_gateway.entry``.  Consume only the two runtime-mode flags here so the
    gateway server sees the same process contract as the Python -> Node ->
    Python launcher path.  Unknown arguments remain available to the caller.
    """

    remaining: list[str] = []
    for arg in argv:
        if arg == "--broker-mode":
            apply_broker_mode_marker(env, enabled=True)
        elif arg == "--qualification-mode":
            apply_qualification_mode_marker(env, enabled=True)
        else:
            remaining.append(arg)
    return remaining
