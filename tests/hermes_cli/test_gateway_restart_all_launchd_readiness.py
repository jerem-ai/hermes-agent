"""Readiness coverage for ``hermes gateway restart --all`` on launchd."""

from types import SimpleNamespace

import pytest

import hermes_cli.gateway as gateway


def _profile(name: str, pid: int):
    return SimpleNamespace(profile=name, pid=pid)


def test_profile_fleet_wait_accepts_reordered_fresh_pids_and_extra_profile(monkeypatch):
    """Fresh replacements are ready regardless of discovery order or additions."""
    monkeypatch.setattr(
        gateway,
        "find_profile_gateway_processes",
        lambda: [_profile("gamma", 55), _profile("beta", 44), _profile("alpha", 33)],
    )

    assert gateway._wait_for_profile_gateway_replacements(
        {"alpha": 11, "beta": 22}, timeout=0
    ) == []


def test_profile_fleet_wait_reports_stale_and_missing_profiles(monkeypatch):
    """An unchanged PID and an absent profile must fail the readiness gate."""
    monkeypatch.setattr(
        gateway,
        "find_profile_gateway_processes",
        lambda: [_profile("alpha", 11), _profile("unrelated", 99)],
    )

    assert gateway._wait_for_profile_gateway_replacements(
        {"alpha": 11, "beta": 22}, timeout=0
    ) == ["alpha", "beta"]


def test_restart_all_waits_for_launchd_named_profile_replacements(monkeypatch, capsys):
    """The command returns only after launchd has recovered its prior fleet."""
    snapshots = iter(
        [
            [_profile("alpha", 11), _profile("beta", 22)],
            [_profile("beta", 44), _profile("alpha", 33)],
        ]
    )
    monkeypatch.setattr(gateway, "_installed_service_kind_for", lambda _platform: "launchd")
    monkeypatch.setattr(gateway, "find_profile_gateway_processes", lambda: next(snapshots))
    monkeypatch.setattr(gateway, "_stop_installed_service", lambda _system: False)
    monkeypatch.setattr(
        gateway,
        "kill_gateway_processes",
        lambda *, all_profiles=False: 2 if all_profiles else pytest.fail("expected fleet kill"),
    )
    monkeypatch.setattr(gateway, "_wait_for_gateway_exit", lambda **_kwargs: True)
    service_calls = []
    monkeypatch.setattr(
        gateway,
        "_service_call",
        lambda kind, verb, system: service_calls.append((kind, verb, system)),
    )

    gateway._restart_all(system=False)

    assert service_calls == [("launchd", "start", False)]
    assert "Stopped 2 gateway process(es) across all profiles" in capsys.readouterr().out
