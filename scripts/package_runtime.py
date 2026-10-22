#!/usr/bin/env python3
"""Package a clean Hermes revision as a read-only runtime artifact."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


MANIFEST_NAME = "hermes-runtime-manifest.json"
LOCK_FILES = ("uv.lock", "package-lock.json")


def _content_digest(entries: list[dict]) -> str:
    encoded = json.dumps(
        sorted(entries, key=lambda item: item["path"].encode("utf-8")),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _archive_content_entry(member: tarfile.TarInfo, payload: bytes | None) -> dict | None:
    if member.isdir():
        return None
    if member.isfile():
        return {
            "path": member.name,
            "kind": "file",
            "executable": bool(member.mode & 0o111),
            "sha256": hashlib.sha256(payload or b"").hexdigest(),
        }
    if member.issym():
        return {
            "path": member.name,
            "kind": "symlink",
            "executable": False,
            "sha256": hashlib.sha256(member.linkname.encode("utf-8")).hexdigest(),
        }
    raise RuntimeError(f"unsupported source entry in runtime artifact: {member.name}")


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_bytes(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
    ).stdout


def _worktree_dirty_against(root: Path, revision: str) -> bool:
    diff = subprocess.run(
        ["git", "-C", str(root), "diff", "--quiet", revision, "--"],
        check=False,
    )
    if diff.returncode not in {0, 1}:
        raise subprocess.CalledProcessError(diff.returncode, diff.args)
    return diff.returncode == 1 or bool(
        _git(root, "ls-files", "--others", "--exclude-standard")
    )


def _lock_digests(root: Path, revision: str) -> dict[str, str]:
    locks: dict[str, str] = {}
    for name in LOCK_FILES:
        exists = subprocess.run(
            ["git", "-C", str(root), "cat-file", "-e", f"{revision}:{name}"],
            check=False,
            capture_output=True,
        )
        if exists.returncode == 0:
            locks[name] = hashlib.sha256(
                _git_bytes(root, "show", f"{revision}:{name}")
            ).hexdigest()
        elif exists.returncode not in {1, 128}:
            raise subprocess.CalledProcessError(exists.returncode, exists.args)
    return locks


def collect_manifest(
    root: Path,
    *,
    python: str | None = None,
    revision: str | None = None,
) -> dict:
    root = root.resolve()
    if Path(_git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise RuntimeError("runtime source must be the exact Git top-level")
    executable = Path(python or sys.executable).resolve()
    resolved_revision = revision or _git(root, "rev-parse", "HEAD^{commit}")
    _git(root, "cat-file", "-e", f"{resolved_revision}^{{commit}}")
    return {
        "contract_version": 1,
        "artifact": {"format": "tar.gz", "filesystem": "read-only"},
        "source": {
            "revision": resolved_revision,
            "tree": _git(root, "rev-parse", f"{resolved_revision}^{{tree}}"),
            "dirty": _worktree_dirty_against(root, resolved_revision),
            "realpath": str(root),
        },
        "locks": _lock_digests(root, resolved_revision),
        "python": {
            "implementation": platform.python_implementation(),
            "realpath": str(executable),
            "version": platform.python_version(),
        },
    }


def _readonly_member(member: tarfile.TarInfo) -> tarfile.TarInfo:
    member.uid = 0
    member.gid = 0
    member.uname = "root"
    member.gname = "root"
    if member.isdir():
        member.mode = 0o555
    elif member.isfile():
        member.mode = 0o555 if member.mode & 0o111 else 0o444
    return member


def _write_git_archive(root: Path, revision: str, destination: Path) -> None:
    with destination.open("wb") as handle:
        subprocess.run(
            ["git", "-C", str(root), "archive", "--format=tar", revision],
            check=True,
            stdout=handle,
        )


def package_runtime(root: Path, output: Path) -> dict:
    root = root.resolve()
    output = output.resolve()
    revision = _git(root, "rev-parse", "HEAD^{commit}")
    manifest = collect_manifest(root, revision=revision)
    if manifest["source"]["dirty"]:
        raise RuntimeError("refusing to package a dirty source tree")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="hermes-runtime-package-") as tmp:
        archive_path = Path(tmp) / "source.tar"
        temp_output = Path(tmp) / "runtime.tar.gz"
        _write_git_archive(root, revision, archive_path)
        with tarfile.open(archive_path, "r:") as source, tarfile.open(
            temp_output, "w:gz", format=tarfile.PAX_FORMAT
        ) as target:
            content_entries: list[dict] = []
            for member in source:
                payload = source.extractfile(member).read() if member.isfile() else None
                entry = _archive_content_entry(member, payload)
                if entry is not None:
                    content_entries.append(entry)
                target.addfile(
                    _readonly_member(member),
                    io.BytesIO(payload) if payload is not None else None,
                )
            manifest["artifact"]["content_sha256"] = _content_digest(content_entries)
            manifest["artifact"]["content_entry_count"] = len(content_entries)
            encoded = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
            manifest_member = tarfile.TarInfo(MANIFEST_NAME)
            manifest_member.size = len(encoded)
            manifest_member.mode = 0o444
            manifest_member.uid = manifest_member.gid = 0
            manifest_member.uname = manifest_member.gname = "root"
            target.addfile(manifest_member, io.BytesIO(encoded))
        os.replace(temp_output, output)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = package_runtime(args.source, args.output)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.error(str(exc))
    artifact_sha256 = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({
        "artifact_sha256": f"sha256:{artifact_sha256}",
        "manifest": manifest,
        "output": str(args.output.resolve()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
