from __future__ import annotations

import json
import hashlib
import subprocess
import types
from pathlib import Path

from hermes_cli import runtime_identity


def test_base_url_classification_catches_private_and_accepts_public_variants():
    assert runtime_identity.classify_base_url("http://127.0.0.1:11434/v1") == "loopback"
    assert runtime_identity.classify_base_url("http://127.1:11434/v1") == "loopback"
    assert runtime_identity.classify_base_url("http://2130706433:11434/v1") == "loopback"
    assert runtime_identity.classify_base_url("http://100.66.155.54:18080/v1") == "tailnet"
    assert runtime_identity.classify_base_url("http://[fd7a:115c:a1e0::1]:18080/v1") == "tailnet"
    assert runtime_identity.classify_base_url(
        "https://model-host.example/v1",
        resolver=lambda _host: ["100.100.10.20"],
    ) == "tailnet"
    assert runtime_identity.classify_base_url("https://models.example.test/v1") == "public"
    assert runtime_identity.classify_base_url(None) == "provider-default"
    assert runtime_identity.classify_base_url("missing-a-scheme") == "invalid"
    assert runtime_identity.classify_base_url("ftp://127.0.0.1/model") == "invalid"
    assert runtime_identity.classify_base_url("http://127.0.0.1:99999/v1") == "invalid"


def test_normalized_origin_fingerprint_is_stable_and_secret_free():
    first = runtime_identity.endpoint_identity(
        "HTTPS://user:secret@Example.COM:443/v1?token=secret#fragment",
        resolver=lambda _host: [],
    )
    second = runtime_identity.endpoint_identity(
        "https://example.com/another/path",
        resolver=lambda _host: [],
    )
    different = runtime_identity.endpoint_identity(
        "https://example.com:8443/v1",
        resolver=lambda _host: [],
    )

    assert first == second
    assert first["base_url_fingerprint"].startswith("sha256:")
    assert first["base_url_fingerprint"] != different["base_url_fingerprint"]
    assert "secret" not in json.dumps(first)


def test_effective_runtime_identity_is_versioned_secret_free_and_broker_unavailable(monkeypatch):
    monkeypatch.setattr(
        runtime_identity,
        "source_identity",
        lambda: {
            "python_realpath": "/runtime/python",
            "source_realpath": "/runtime/hermes",
            "source_revision": "a" * 40,
            "source_tree": "b" * 40,
            "source_dirty": False,
                "source_identity_provenance": "embedded-manifest",
                "source_content_sha256": "sha256:" + "c" * 64,
        },
    )
    monkeypatch.setattr(runtime_identity, "broker_mode_requested", lambda: True)
    monkeypatch.setattr(runtime_identity, "qualification_mode_requested", lambda: True)
    raw_url = "https://token-value@private.example:9443/secret-path"

    result = runtime_identity.effective_runtime_identity(
        types.SimpleNamespace(provider="custom", base_url=raw_url)
    )

    assert result["contract_version"] == 1
    assert result["provider"] == "custom"
    assert result["base_url_class"] == "public"
    assert result["broker"] == {"contract_version": 1, "mode": True, "available": False}
    assert result["qualification"] == {
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
    assert raw_url not in json.dumps(result)
    assert "token-value" not in json.dumps(result)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _init_parent_repo(root: Path) -> None:
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Hermes Test")
    _git(root, "config", "user.email", "hermes@example.invalid")
    (root / "parent.txt").write_text("parent\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "parent")


def test_capture_source_identity_rejects_parent_repo_git_identity(tmp_path):
    _init_parent_repo(tmp_path)
    nested = tmp_path / "installed" / "hermes"
    nested.mkdir(parents=True)

    captured = runtime_identity.capture_source_identity(nested, Path("/runtime/python"))

    assert captured["source_identity_provenance"] == "unverified"
    assert captured["source_revision"] is None
    assert captured["source_tree"] is None


def test_embedded_manifest_wins_over_parent_repo_and_validates_lock(tmp_path):
    _init_parent_repo(tmp_path)
    nested = tmp_path / "installed" / "hermes"
    nested.mkdir(parents=True)
    lock = nested / "uv.lock"
    lock.write_text("locked\n", encoding="utf-8")
    lock.chmod(0o444)
    nested.chmod(0o555)
    revision = "a" * 40
    tree = "b" * 40
    content = runtime_identity._installed_content_digest(nested)
    assert content is not None
    manifest = {
        "contract_version": 1,
        "artifact": {
            "format": "tar.gz",
            "filesystem": "read-only",
            "content_sha256": content[0],
            "content_entry_count": content[1],
        },
        "source": {"revision": revision, "tree": tree, "dirty": False},
        "locks": {"uv.lock": hashlib.sha256(lock.read_bytes()).hexdigest()},
    }
    nested.chmod(0o755)
    (nested / runtime_identity._PACKAGED_MANIFEST).write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (nested / runtime_identity._PACKAGED_MANIFEST).chmod(0o444)
    nested.chmod(0o555)

    captured = runtime_identity.capture_source_identity(nested, Path("/runtime/python"))

    assert captured["source_identity_provenance"] == "embedded-manifest"
    assert captured["source_revision"] == revision
    assert captured["source_tree"] == tree
    assert captured["source_content_sha256"] == content[0]

    lock.chmod(0o644)
    writable_file = runtime_identity.capture_source_identity(
        nested, Path("/runtime/python")
    )
    assert writable_file["source_identity_provenance"] == "invalid-embedded-manifest"
    assert writable_file["source_revision"] is None
    lock.chmod(0o444)

    nested.chmod(0o755)
    writable_root = runtime_identity.capture_source_identity(
        nested, Path("/runtime/python")
    )
    assert writable_root["source_identity_provenance"] == "invalid-embedded-manifest"
    assert writable_root["source_revision"] is None
    nested.chmod(0o555)

    manifest["locks"]["uv.lock"] = "0" * 64
    (nested / runtime_identity._PACKAGED_MANIFEST).chmod(0o644)
    (nested / runtime_identity._PACKAGED_MANIFEST).write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (nested / runtime_identity._PACKAGED_MANIFEST).chmod(0o444)
    invalid = runtime_identity.capture_source_identity(nested, Path("/runtime/python"))
    assert invalid["source_identity_provenance"] == "invalid-embedded-manifest"
    assert invalid["source_revision"] is None


def _packaged_tree(root: Path) -> tuple[Path, str]:
    nested = root / "installed" / "hermes"
    (nested / "pkg" / "ünïcode").mkdir(parents=True)
    (nested / "pkg" / "ünïcode" / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    (nested / "pkg" / "\U0001d521-astral.txt").write_text("astral\n", encoding="utf-8")
    (nested / "pkg" / "Ａ-wide.txt").write_text("wide\n", encoding="utf-8")
    lock = nested / "uv.lock"
    lock.write_text("locked\n", encoding="utf-8")
    for path in sorted(nested.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    nested.chmod(0o555)
    content = runtime_identity._installed_content_digest(nested)
    assert content is not None
    manifest = {
        "contract_version": 1,
        "artifact": {
            "format": "tar.gz",
            "filesystem": "read-only",
            "content_sha256": content[0],
            "content_entry_count": content[1],
        },
        "source": {"revision": "a" * 40, "tree": "b" * 40, "dirty": False},
        "locks": {"uv.lock": hashlib.sha256(lock.read_bytes()).hexdigest()},
    }
    nested.chmod(0o755)
    (nested / runtime_identity._PACKAGED_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    (nested / runtime_identity._PACKAGED_MANIFEST).chmod(0o444)
    nested.chmod(0o555)
    return nested, content[0]


def _launcher_claim(monkeypatch, content_sha256: str, fingerprint: str) -> None:
    monkeypatch.setenv(runtime_identity._LAUNCHER_SOURCE_SHA256_ENV, content_sha256)
    monkeypatch.setenv(runtime_identity._LAUNCHER_SOURCE_FINGERPRINT_ENV, fingerprint)


def test_launcher_fingerprint_skips_the_content_walk_for_the_tree_it_verified(tmp_path, monkeypatch):
    nested, content = _packaged_tree(tmp_path)
    _launcher_claim(monkeypatch, content, runtime_identity._installed_stat_fingerprint(nested))

    def no_walk(*_args, **_kwargs):
        raise AssertionError("contents were hashed again")

    monkeypatch.setattr(runtime_identity, "_installed_content_digest", no_walk)
    captured = runtime_identity.capture_source_identity(nested, Path("/runtime/python"))

    assert captured["source_identity_provenance"] == "embedded-manifest"
    assert captured["source_content_sha256"] == content
    assert captured["source_content_verified_by"] == "launcher"


def test_a_file_changed_after_the_launcher_verified_it_is_caught(tmp_path, monkeypatch):
    nested, content = _packaged_tree(tmp_path)
    _launcher_claim(monkeypatch, content, runtime_identity._installed_stat_fingerprint(nested))
    module = nested / "pkg" / "ünïcode" / "module.py"
    module.chmod(0o644)
    module.write_text("VALUE = 2\n", encoding="utf-8")
    module.chmod(0o444)

    captured = runtime_identity.capture_source_identity(nested, Path("/runtime/python"))

    assert captured["source_identity_provenance"] == "invalid-embedded-manifest"
    assert captured["source_content_sha256"] is None


def test_a_launcher_claim_for_other_contents_falls_back_to_hashing(tmp_path, monkeypatch):
    nested, content = _packaged_tree(tmp_path)
    _launcher_claim(monkeypatch, "sha256:" + "0" * 64, runtime_identity._installed_stat_fingerprint(nested))

    captured = runtime_identity.capture_source_identity(nested, Path("/runtime/python"))

    assert captured["source_content_sha256"] == content
    assert captured["source_content_verified_by"] == "hermes"

    _launcher_claim(monkeypatch, content, "not-a-fingerprint")
    assert runtime_identity.capture_source_identity(
        nested, Path("/runtime/python")
    )["source_content_verified_by"] == "hermes"


def test_stat_fingerprint_matches_the_launcher_line_format(tmp_path):
    nested, _content = _packaged_tree(tmp_path)
    root = nested.resolve()
    expected = hashlib.sha256()

    def entry(path: Path, relative: str | None) -> None:
        metadata = path.lstat()
        if relative is None:
            expected.update(
                f"{path}\0{metadata.st_mode}\0{metadata.st_ino}\0{metadata.st_mtime_ns}\0{metadata.st_ctime_ns}\n".encode()
            )
        else:
            expected.update(
                f"{relative}\0{metadata.st_mode}\0{metadata.st_size}\0{metadata.st_ino}"
                f"\0{metadata.st_mtime_ns}\0{metadata.st_ctime_ns}\n".encode()
            )

    entry(root, None)
    # JavaScript sorts names by UTF-16 code unit, which puts the astral
    # U+1D521 (surrogate 0xD835) before U+FF21; code point order would not.
    for relative in [
        "hermes-runtime-manifest.json",
        "pkg",
        "pkg/ünïcode",
        "pkg/ünïcode/module.py",
        "pkg/\U0001d521-astral.txt",
        "pkg/Ａ-wide.txt",
        "uv.lock",
    ]:
        entry(root / relative, relative)

    assert runtime_identity._installed_stat_fingerprint(root) == f"sha256:{expected.hexdigest()}"


def test_source_identity_is_bootstrap_snapshot(monkeypatch):
    before = runtime_identity.source_identity()
    monkeypatch.setattr(runtime_identity, "capture_source_identity", lambda *_args: {"changed": True})

    assert runtime_identity.source_identity() == before


def test_git_identity_tree_and_dirty_state_stay_bound_to_captured_revision(tmp_path, monkeypatch):
    root = tmp_path / "runtime"
    root.mkdir()
    _init_parent_repo(root)
    first_revision = _git(root, "rev-parse", "HEAD")
    first_tree = _git(root, "rev-parse", f"{first_revision}^{{tree}}")
    (root / "parent.txt").write_text("second\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "second")
    second_revision = _git(root, "rev-parse", "HEAD")
    _git(root, "checkout", "-q", first_revision)
    original = runtime_identity._run_git

    def advance_after_revision(source_root, *args):
        result = original(source_root, *args)
        if args == ("rev-parse", "HEAD^{commit}"):
            _git(root, "checkout", "-q", second_revision)
        return result

    monkeypatch.setattr(runtime_identity, "_run_git", advance_after_revision)
    captured = runtime_identity.capture_source_identity(root, Path("/runtime/python"))

    assert captured["source_revision"] == first_revision
    assert captured["source_tree"] == first_tree
    assert captured["source_dirty"] is True
