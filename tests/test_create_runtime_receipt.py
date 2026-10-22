from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


package_runtime = _load("package_runtime_for_receipt", ROOT / "scripts" / "package_runtime.py")
create_runtime_receipt = _load("create_runtime_receipt", ROOT / "scripts" / "create_runtime_receipt.py")


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def _installed_fixture(tmp_path: Path, source_text: str = "print('ok')\n"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    build = tmp_path / "build 雪"
    build.mkdir()
    _git(build, "init", "-q")
    _git(build, "config", "user.name", "Receipt Test")
    _git(build, "config", "user.email", "receipt@example.invalid")
    (build / "uv.lock").write_text("lock\n", encoding="utf-8")
    (build / "run_agent.py").write_text(source_text, encoding="utf-8")
    _git(build, "add", ".")
    _git(build, "commit", "-qm", "fixture")
    artifact = tmp_path / "runtime.tar.gz"
    package_runtime.package_runtime(build, artifact)
    runtime = tmp_path / "runtime Ω"
    runtime.mkdir()
    installed = runtime / "source"
    installed.mkdir()
    with tarfile.open(artifact, "r:gz") as archive:
        archive.extractall(installed)
    installed.chmod(0o555)
    environment = runtime / "venv"
    python_bin = environment / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_runtime = runtime / "python-runtime"
    python_target = python_runtime / "bin" / "python3.11"
    python_target.parent.mkdir(parents=True)
    python_target.write_text("#!/bin/sh\n", encoding="utf-8")
    python_target.chmod(0o555)
    python_bin.symlink_to(python_target)
    (python_bin.parent / "python3").symlink_to("python")
    python_bin.parent.chmod(0o555)
    environment.chmod(0o555)
    python_target.parent.chmod(0o555)
    python_runtime.chmod(0o555)
    runtime.chmod(0o555)
    return artifact, runtime, installed, python_bin


def test_create_receipt_binds_unicode_install_artifact_and_content(tmp_path):
    artifact, runtime, installed, python_bin = _installed_fixture(tmp_path)
    output = tmp_path / "receipts" / "receipt.json"

    result = create_runtime_receipt.create_receipt(
        artifact, runtime, installed, runtime / "python-runtime", python_bin, output
    )

    assert result["receipt"]["agentRoot"] == str(installed.resolve())
    assert result["receipt"]["pythonBin"] == str(python_bin)
    assert result["receipt"]["pythonRealpath"] == str(python_bin.resolve())
    assert result["receipt"]["sourceContentSha256"].startswith("sha256:")
    assert result["receipt"]["artifactSha256"].startswith("sha256:")
    assert result["receiptSha256"].startswith("sha256:")
    assert output.stat().st_mode & 0o777 == 0o400


def test_create_receipt_rejects_an_environment_symlink_escape(tmp_path):
    artifact, runtime, installed, python_bin = _installed_fixture(tmp_path)
    runtime.chmod(0o755)
    environment = python_bin.parent.parent
    environment.chmod(0o755)
    packages = environment / "lib"
    packages.mkdir()
    outside = tmp_path / "mutable-dependency.py"
    outside.write_text("VALUE = 1\n", encoding="utf-8")
    (packages / "escape.py").symlink_to(outside)
    packages.chmod(0o555)
    environment.chmod(0o555)
    runtime.chmod(0o555)

    try:
        create_runtime_receipt.create_receipt(
            artifact, runtime, installed, runtime / "python-runtime", python_bin,
            tmp_path / "receipt.json"
        )
    except RuntimeError as exc:
        assert "content-addressable" in str(exc)
    else:
        raise AssertionError("external dependency symlink should be refused")


def test_create_receipt_rejects_external_or_executable_python_path_files(tmp_path):
    for line in (str(tmp_path / "external modules"), "import external_finder"):
        case = tmp_path / hashlib.sha256(line.encode()).hexdigest()[:8]
        artifact, runtime, installed, python_bin = _installed_fixture(case)
        runtime.chmod(0o755)
        environment = python_bin.parent.parent
        environment.chmod(0o755)
        site_packages = environment / "lib" / "python3.11" / "site-packages"
        site_packages.mkdir(parents=True)
        path_file = site_packages / "escape.pth"
        path_file.write_text(f"{line}\n", encoding="utf-8")
        path_file.chmod(0o444)
        for directory in (environment / "lib", environment / "lib" / "python3.11", site_packages):
            directory.chmod(0o555)
        environment.chmod(0o555)
        runtime.chmod(0o555)

        with pytest.raises(RuntimeError, match="content-addressable"):
            create_runtime_receipt.create_receipt(
                artifact, runtime, installed, runtime / "python-runtime", python_bin,
                case / "receipt.json"
            )


def test_create_receipt_accepts_an_internal_unicode_python_path(tmp_path):
    artifact, runtime, installed, python_bin = _installed_fixture(tmp_path)
    runtime.chmod(0o755)
    environment = python_bin.parent.parent
    environment.chmod(0o755)
    python_lib = environment / "lib" / "python3.11"
    site_packages = python_lib / "site-packages"
    internal = python_lib / "内部 Ω"
    site_packages.mkdir(parents=True)
    internal.mkdir()
    path_file = site_packages / "internal.pth"
    path_file.write_text("../内部 Ω\n", encoding="utf-8")
    path_file.chmod(0o444)
    for directory in (environment / "lib", python_lib, site_packages, internal):
        directory.chmod(0o555)
    environment.chmod(0o555)
    runtime.chmod(0o555)

    result = create_runtime_receipt.create_receipt(
        artifact, runtime, installed, runtime / "python-runtime", python_bin,
        tmp_path / "receipt.json"
    )
    assert result["receipt"]["pythonEnvironmentEntryCount"] >= 3


def test_create_receipt_canonicalizes_an_aliased_runtime_entry_path(tmp_path):
    artifact, runtime, installed, _python_bin = _installed_fixture(tmp_path / "fixture")
    alias = tmp_path / "runtime-alias"
    alias.symlink_to(runtime, target_is_directory=True)

    result = create_runtime_receipt.create_receipt(
        artifact, runtime, installed, runtime / "python-runtime",
        alias / "venv" / "bin" / "python", tmp_path / "receipt.json"
    )

    assert result["receipt"]["pythonBin"] == str(runtime / "venv" / "bin" / "python")


def test_create_receipt_rejects_a_source_path_delimiter(tmp_path):
    artifact, runtime, installed, python_bin = _installed_fixture(tmp_path / "path:split")

    with pytest.raises(RuntimeError, match="Python path delimiter"):
        create_runtime_receipt.create_receipt(
            artifact, runtime, installed, runtime / "python-runtime", python_bin,
            tmp_path / "receipt.json"
        )


def test_frozen_python_runtime_creates_an_operable_receipted_virtualenv(tmp_path):
    artifact, runtime, installed, _fixture_python = _installed_fixture(tmp_path / "fixture")
    runtime.chmod(0o755)
    fixture_python_runtime = runtime / "python-runtime"
    for candidate in sorted(fixture_python_runtime.rglob("*"), reverse=True):
        if candidate.is_dir():
            candidate.chmod(0o755)
        elif not candidate.is_symlink():
            candidate.chmod(0o644)
    fixture_python_runtime.chmod(0o755)
    shutil.rmtree(fixture_python_runtime)
    uv_roots = sorted(
        root for root in (Path.home() / ".local" / "share" / "uv" / "python").glob("cpython-3.11*")
        if (root / "bin" / "python3.11").is_file()
    )
    if not uv_roots:
        pytest.skip("a relocatable uv-managed CPython 3.11 runtime is required for this install smoke")
    source_python_runtime = uv_roots[-1]
    shutil.copytree(source_python_runtime, fixture_python_runtime, symlinks=True)
    for path_file in fixture_python_runtime.rglob("*.pth"):
        path_file.unlink()
    environment = runtime / "venv"
    for candidate in sorted(environment.rglob("*"), reverse=True):
        if candidate.is_dir():
            candidate.chmod(0o755)
        elif not candidate.is_symlink():
            candidate.chmod(0o644)
    environment.chmod(0o755)
    shutil.rmtree(environment)
    frozen_python = fixture_python_runtime / "bin" / "python3.11"
    subprocess.run([frozen_python, "-m", "venv", "--without-pip", environment], check=True)
    for name in ("python", "python3", "python3.11"):
        entry = environment / "bin" / name
        entry.unlink(missing_ok=True)
        entry.symlink_to(frozen_python)
    python_bin = environment / "bin" / "python"
    assert python_bin.is_symlink()
    result = subprocess.run(
        [python_bin, "-c", (
            "import json,sys,sysconfig; "
            "print(json.dumps({'ok': True, 'prefix': sys.prefix, "
            "'base_prefix': sys.base_prefix, 'stdlib': sysconfig.get_path('stdlib')}))"
        )],
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(result.stdout)
    assert observed["ok"] is True
    assert Path(observed["base_prefix"]).resolve() == fixture_python_runtime.resolve()
    assert fixture_python_runtime.resolve() in Path(observed["stdlib"]).resolve().parents
    for root in (fixture_python_runtime, environment):
        for candidate in sorted(root.rglob("*"), reverse=True):
            if candidate.is_symlink():
                continue
            candidate.chmod(0o555 if candidate.is_dir() or candidate.stat().st_mode & 0o111 else 0o444)
        root.chmod(0o555)
    runtime.chmod(0o555)
    receipt = create_runtime_receipt.create_receipt(
        artifact, runtime, installed, fixture_python_runtime, python_bin,
        tmp_path / "real-receipt.json"
    )
    assert receipt["receipt"]["pythonRealpath"] == str(python_bin.resolve())


def test_create_receipt_rejects_a_writable_external_python_target(tmp_path):
    artifact, runtime, installed, python_bin = _installed_fixture(tmp_path)
    python_bin.resolve().chmod(0o755)

    try:
        create_runtime_receipt.create_receipt(
            artifact, runtime, installed, runtime / "python-runtime", python_bin,
            tmp_path / "receipt.json"
        )
    except RuntimeError as exc:
        assert "inside the frozen Python runtime" in str(exc)
    else:
        raise AssertionError("writable external Python target should be refused")


def test_create_receipt_rejects_modified_installed_source(tmp_path):
    artifact, runtime, installed, python_bin = _installed_fixture(tmp_path)
    (installed / "run_agent.py").chmod(0o644)
    (installed / "run_agent.py").write_text("print('changed')\n", encoding="utf-8")

    try:
        create_runtime_receipt.create_receipt(
            artifact, runtime, installed, runtime / "python-runtime", python_bin,
            tmp_path / "receipt.json"
        )
    except RuntimeError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("modified source should be refused")


def test_create_receipt_rejects_writable_installed_file_or_root(tmp_path):
    artifact, runtime, installed, python_bin = _installed_fixture(tmp_path)
    (installed / "run_agent.py").chmod(0o644)
    try:
        create_runtime_receipt.create_receipt(
            artifact, runtime, installed, runtime / "python-runtime", python_bin,
            tmp_path / "file-receipt.json"
        )
    except RuntimeError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("writable source file should be refused")

    (installed / "run_agent.py").chmod(0o444)
    installed.chmod(0o755)
    try:
        create_runtime_receipt.create_receipt(
            artifact, runtime, installed, runtime / "python-runtime", python_bin,
            tmp_path / "root-receipt.json"
        )
    except RuntimeError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("writable source root should be refused")


def test_create_receipt_rejects_artifact_from_a_different_clean_runtime(tmp_path):
    artifact_a, _runtime_a, _installed_a, _python_a = _installed_fixture(tmp_path / "a")
    artifact_b, runtime_b, installed_b, python_b = _installed_fixture(tmp_path / "b", "print('different')\n")
    del artifact_b

    try:
        create_runtime_receipt.create_receipt(
            artifact_a, runtime_b, installed_b, runtime_b / "python-runtime", python_b,
            tmp_path / "receipt.json"
        )
    except RuntimeError as exc:
        assert "do not match" in str(exc)
    else:
        raise AssertionError("mismatched artifact and installation should be refused")


def test_create_receipt_hash_and_identity_use_one_artifact_snapshot(tmp_path, monkeypatch):
    artifact_a, runtime_a, installed_a, python_a = _installed_fixture(tmp_path / "a")
    artifact_b, _runtime_b, _installed_b, _python_b = _installed_fixture(tmp_path / "b", "print('different')\n")
    original_bytes = artifact_a.read_bytes()
    original_identity = create_runtime_receipt._artifact_identity

    def replace_after_validation(artifact_bytes):
        identity = original_identity(artifact_bytes)
        os.replace(artifact_b, artifact_a)
        return identity

    monkeypatch.setattr(create_runtime_receipt, "_artifact_identity", replace_after_validation)
    result = create_runtime_receipt.create_receipt(
        artifact_a, runtime_a, installed_a, runtime_a / "python-runtime", python_a,
        tmp_path / "receipt.json"
    )

    assert result["receipt"]["artifactSha256"] == (
        f"sha256:{hashlib.sha256(original_bytes).hexdigest()}"
    )
    assert result["receipt"]["artifactSha256"] != create_runtime_receipt.sha256(artifact_a)


def test_artifact_identity_rejects_writable_file_and_directory_members():
    for name, mode, kind in (
        ("writable.txt", 0o644, "file"),
        ("writable-dir", 0o755, "directory"),
    ):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
            manifest = json.dumps({}).encode("utf-8")
            manifest_member = tarfile.TarInfo("hermes-runtime-manifest.json")
            manifest_member.size = len(manifest)
            manifest_member.mode = 0o444
            archive.addfile(manifest_member, io.BytesIO(manifest))
            candidate = tarfile.TarInfo(name)
            candidate.mode = mode
            if kind == "directory":
                candidate.type = tarfile.DIRTYPE
                archive.addfile(candidate)
            else:
                candidate.size = 1
                archive.addfile(candidate, io.BytesIO(b"x"))

        try:
            create_runtime_receipt._artifact_identity(buffer.getvalue())
        except RuntimeError as exc:
            assert "writable content" in str(exc)
        else:
            raise AssertionError(f"writable artifact {kind} should be refused")
