from __future__ import annotations

import importlib.util
import json
import subprocess
import tarfile
from pathlib import Path


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "package_runtime.py"
_SPEC = importlib.util.spec_from_file_location("package_runtime", _SCRIPT)
assert _SPEC and _SPEC.loader
package_runtime = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(package_runtime)


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "Hermes source with ünicode"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Hermes Test")
    _git(root, "config", "user.email", "hermes@example.invalid")
    (root / "uv.lock").write_text("lock fixture\n", encoding="utf-8")
    executable = root / "launch.sh"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "fixture")
    return root


def test_package_runtime_refuses_dirty_source(tmp_path):
    root = _repo(tmp_path)
    (root / "untracked.txt").write_text("dirty\n", encoding="utf-8")

    try:
        package_runtime.package_runtime(root, tmp_path / "runtime.tar.gz")
    except RuntimeError as exc:
        assert "dirty source tree" in str(exc)
    else:
        raise AssertionError("dirty source should be refused")


def test_package_runtime_rejects_nested_directory_inside_parent_repo(tmp_path):
    root = _repo(tmp_path)
    nested = root / "nested-runtime"
    nested.mkdir()
    (nested / "entry.py").write_text("print('nested')\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "nested")

    try:
        package_runtime.package_runtime(nested, tmp_path / "runtime.tar.gz")
    except RuntimeError as exc:
        assert "exact Git top-level" in str(exc)
    else:
        raise AssertionError("nested source should be refused")


def test_package_runtime_accepts_unicode_path_and_emits_read_only_identity(tmp_path):
    root = _repo(tmp_path)
    output = tmp_path / "runtime.tar.gz"

    manifest = package_runtime.package_runtime(root, output)

    assert manifest["source"]["dirty"] is False
    assert manifest["source"]["realpath"] == str(root.resolve())
    assert len(manifest["source"]["revision"]) == 40
    assert len(manifest["source"]["tree"]) == 40
    assert len(manifest["locks"]["uv.lock"]) == 64
    assert manifest["artifact"]["content_sha256"].startswith("sha256:")
    assert manifest["artifact"]["content_entry_count"] == 2
    with tarfile.open(output, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()}
        embedded = json.load(archive.extractfile(package_runtime.MANIFEST_NAME))
    assert embedded == manifest
    assert members["uv.lock"].mode == 0o444
    assert members["launch.sh"].mode == 0o555
    assert members[package_runtime.MANIFEST_NAME].mode == 0o444


def test_package_runtime_preserves_nested_tree_as_read_only(tmp_path):
    root = _repo(tmp_path)
    nested = root / "nested ü"
    nested.mkdir()
    (nested / "data.txt").write_text("nested\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "nested read-only fixture")
    output = tmp_path / "nested-runtime.tar.gz"

    package_runtime.package_runtime(root, output)

    with tarfile.open(output, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()}
    assert members["nested ü"].mode == 0o555
    assert members["nested ü/data.txt"].mode == 0o444


def test_package_runtime_binds_manifest_and_archive_to_one_oid_during_head_race(tmp_path, monkeypatch):
    root = _repo(tmp_path)
    first_revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    (root / "uv.lock").write_text("second lock\n", encoding="utf-8")
    _git(root, "add", "uv.lock")
    _git(root, "commit", "-qm", "second")
    second_revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _git(root, "checkout", "-q", first_revision)
    original_write = package_runtime._write_git_archive

    def move_head_then_archive(source, revision, destination):
        _git(source, "update-ref", "HEAD", second_revision)
        original_write(source, revision, destination)

    monkeypatch.setattr(package_runtime, "_write_git_archive", move_head_then_archive)
    output = tmp_path / "raced-runtime.tar.gz"

    manifest = package_runtime.package_runtime(root, output)

    with tarfile.open(output, "r:gz") as archive:
        archived_lock = archive.extractfile("uv.lock").read().decode()
        embedded = json.load(archive.extractfile(package_runtime.MANIFEST_NAME))
    assert manifest["source"]["revision"] == first_revision
    assert embedded["source"]["revision"] == first_revision
    assert archived_lock == "lock fixture\n"


def test_packaged_content_digest_changes_when_source_bytes_change(tmp_path):
    root = _repo(tmp_path)
    output = tmp_path / "runtime.tar.gz"
    manifest = package_runtime.package_runtime(root, output)
    extracted = tmp_path / "installed"
    with tarfile.open(output, "r:gz") as archive:
        archive.extractall(extracted)
    extracted.chmod(0o555)

    from hermes_cli.runtime_identity import capture_source_identity

    valid = capture_source_identity(extracted, Path("/runtime/python"))
    assert valid["source_content_sha256"] == manifest["artifact"]["content_sha256"]
    (extracted / "launch.sh").chmod(0o755)
    (extracted / "launch.sh").write_text("#!/bin/sh\nexit 99\n", encoding="utf-8")
    altered = capture_source_identity(extracted, Path("/runtime/python"))
    assert altered["source_identity_provenance"] == "invalid-embedded-manifest"
    assert altered["source_content_sha256"] is None
