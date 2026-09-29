"""launchd plist refresh must not report success when launchd never registered the service (#12866)."""
import subprocess
from unittest.mock import MagicMock

import pytest

from hermes_cli import gateway as gw


def _stale_plist(tmp_path, monkeypatch, *, registered: bool):
    plist_path = tmp_path / "com.hermes.plist"
    plist_path.write_text("<old/>", encoding="utf-8")
    monkeypatch.setattr(gw, "get_launchd_plist_path", lambda: plist_path)
    monkeypatch.setattr(gw, "launchd_plist_is_current", lambda: False)
    monkeypatch.setattr(gw, "generate_launchd_plist", lambda: "<new/>")
    monkeypatch.setattr(gw, "_refuse_temp_home_service_write", lambda *a: False)
    monkeypatch.setattr(gw, "get_launchd_label", lambda: "com.hermes.agent")
    monkeypatch.setattr(gw, "_launchd_domain", lambda: "gui/501")
    monkeypatch.setattr(gw, "_append_launchd_reload_log", lambda msg: None)
    monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
    monkeypatch.setattr(gw, "_retry_launchctl_bootstrap_until_registered", lambda *a, **k: registered)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: MagicMock(returncode=0))


def test_refresh_reports_registration_outcome(tmp_path, monkeypatch, capsys):
    _stale_plist(tmp_path, monkeypatch, registered=False)
    assert gw.refresh_launchd_plist_if_needed() is False
    assert "Updated" not in capsys.readouterr().out

    _stale_plist(tmp_path, monkeypatch, registered=True)
    assert gw.refresh_launchd_plist_if_needed() is True
    assert "Updated" in capsys.readouterr().out


def test_launchd_start_waits_for_replacement_after_deferred_refresh(tmp_path, monkeypatch, capsys):
    """A stale-plist reload must settle before start reports success."""
    plist_path = tmp_path / "ai.hermes.gateway.plist"
    plist_path.write_text("<old/>", encoding="utf-8")
    waited = []

    monkeypatch.setattr(gw, "get_launchd_plist_path", lambda: plist_path)
    monkeypatch.setattr(gw, "get_launchd_label", lambda: "ai.hermes.gateway")
    monkeypatch.setattr(gw, "_launchd_domain", lambda: "gui/501")
    monkeypatch.setattr("gateway.status.get_running_pid", lambda **_kwargs: 41)
    monkeypatch.setattr(gw, "refresh_launchd_plist_if_needed", lambda: True)
    monkeypatch.setattr(gw, "_launchd_reload_budget", lambda: 12.0)
    monkeypatch.setattr(
        gw,
        "_wait_for_launchd_service_pid",
        lambda label, old_pid, timeout, domain: waited.append(
            (label, old_pid, timeout, domain)
        ) or True,
    )
    monkeypatch.setattr(
        gw,
        "_launchctl_kickstart_current",
        lambda _label: pytest.fail("must not kickstart the job a deferred helper will replace"),
    )

    gw.launchd_start()

    assert waited == [("ai.hermes.gateway", 41, 17.0, "gui/501")]
    assert "✓ Service started" in capsys.readouterr().out


def test_launchd_start_fails_when_refreshed_service_has_no_replacement(tmp_path, monkeypatch, capsys):
    """A helper exit without a supervised replacement is a failed start."""
    plist_path = tmp_path / "ai.hermes.gateway.plist"
    plist_path.write_text("<old/>", encoding="utf-8")

    monkeypatch.setattr(gw, "get_launchd_plist_path", lambda: plist_path)
    monkeypatch.setattr(gw, "get_launchd_label", lambda: "ai.hermes.gateway")
    monkeypatch.setattr(gw, "_launchd_domain", lambda: "gui/501")
    monkeypatch.setattr("gateway.status.get_running_pid", lambda **_kwargs: 41)
    monkeypatch.setattr(gw, "refresh_launchd_plist_if_needed", lambda: True)
    monkeypatch.setattr(gw, "_launchd_reload_budget", lambda: 0.0)
    monkeypatch.setattr(gw, "_wait_for_launchd_service_pid", lambda *args, **kwargs: False)

    with pytest.raises(SystemExit) as exc:
        gw.launchd_start()

    assert exc.value.code == 1
    assert "did not supervise a replacement process" in capsys.readouterr().out


def test_launchd_start_unchanged_plist_uses_normal_kickstart(tmp_path, monkeypatch, capsys):
    """An already-current service keeps the direct start path."""
    plist_path = tmp_path / "ai.hermes.gateway.plist"
    plist_path.write_text("<current/>", encoding="utf-8")
    kicked = []

    monkeypatch.setattr(gw, "get_launchd_plist_path", lambda: plist_path)
    monkeypatch.setattr(gw, "get_launchd_label", lambda: "ai.hermes.gateway")
    monkeypatch.setattr("gateway.status.get_running_pid", lambda **_kwargs: None)
    monkeypatch.setattr(gw, "refresh_launchd_plist_if_needed", lambda: False)
    monkeypatch.setattr(gw, "_launchctl_kickstart_current", kicked.append)

    gw.launchd_start()

    assert kicked == ["ai.hermes.gateway"]
    assert "✓ Service started" in capsys.readouterr().out


def test_launchd_stop_waits_until_bootout_registration_is_gone(monkeypatch, capsys):
    """A completed process exit is insufficient while launchd still owns the label."""
    waited = []

    monkeypatch.setattr(gw, "_launchd_domain", lambda: "user/501")
    monkeypatch.setattr(gw, "get_launchd_label", lambda: "ai.hermes.gateway")
    monkeypatch.setattr(gw, "_mark_planned_stop", lambda: None)
    monkeypatch.setattr(
        gw.subprocess,
        "run",
        lambda *args, **kwargs: MagicMock(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(gw, "_wait_for_gateway_exit", lambda **kwargs: True)
    monkeypatch.setattr(
        gw,
        "_wait_for_launchd_service_unloaded",
        lambda domain, label, timeout: waited.append((domain, label, timeout)) or True,
    )

    gw.launchd_stop()

    assert waited == [("user/501", "ai.hermes.gateway", 10.0)]
    assert "✓ Service stopped" in capsys.readouterr().out
