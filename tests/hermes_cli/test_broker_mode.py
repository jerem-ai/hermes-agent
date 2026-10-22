from __future__ import annotations

import os

from hermes_cli.broker_mode import (
    _BROKER_MODE_MARKER,
    _QUALIFICATION_MODE_MARKER,
    apply_broker_mode_marker,
    apply_qualification_mode_marker,
    broker_mode_requested,
    qualification_mode_requested,
)
from hermes_cli.env_loader import load_hermes_dotenv


def test_broker_mode_skips_installation_env_but_keeps_scoped_home(tmp_path, monkeypatch):
    home = tmp_path / "profile-home"
    home.mkdir()
    (home / ".env").write_text("BROKER_HOME_VALUE=scoped\n", encoding="utf-8")
    project_env = tmp_path / "installation.env"
    project_env.write_text("BROKER_INSTALL_VALUE=must-not-load\n", encoding="utf-8")
    monkeypatch.setenv(_BROKER_MODE_MARKER, "1")
    monkeypatch.delenv("BROKER_HOME_VALUE", raising=False)
    monkeypatch.delenv("BROKER_INSTALL_VALUE", raising=False)

    loaded = load_hermes_dotenv(
        hermes_home=home,
        project_env=project_env,
        load_external_secrets=False,
    )

    assert loaded == [home / ".env"]
    assert os.environ["BROKER_HOME_VALUE"] == "scoped"
    assert "BROKER_INSTALL_VALUE" not in os.environ


def test_normal_mode_retains_installation_env_fallback(tmp_path, monkeypatch):
    project_env = tmp_path / "installation.env"
    project_env.write_text("BROKER_NORMAL_VALUE=loaded\n", encoding="utf-8")
    monkeypatch.delenv(_BROKER_MODE_MARKER, raising=False)
    monkeypatch.delenv("BROKER_NORMAL_VALUE", raising=False)

    loaded = load_hermes_dotenv(
        hermes_home=tmp_path / "missing-home",
        project_env=project_env,
        load_external_secrets=False,
    )

    assert loaded == [project_env]
    assert os.environ["BROKER_NORMAL_VALUE"] == "loaded"


def test_broker_mode_marker_is_explicit_and_clears_stale_state():
    env = {_BROKER_MODE_MARKER: "1"}
    apply_broker_mode_marker(env, enabled=False)
    assert not broker_mode_requested(argv=[], environ=env)

    apply_broker_mode_marker(env, enabled=True)
    assert broker_mode_requested(argv=[], environ=env)
    assert broker_mode_requested(argv=["--broker-mode"], environ={})


def test_qualification_mode_marker_survives_child_projection_and_clears_stale_state():
    env = {_QUALIFICATION_MODE_MARKER: "1"}
    apply_qualification_mode_marker(env, enabled=False)
    assert not qualification_mode_requested(argv=[], environ=env)

    apply_qualification_mode_marker(env, enabled=True)
    assert qualification_mode_requested(argv=[], environ=env)
    assert qualification_mode_requested(argv=["--qualification-mode"], environ={})
