#!/usr/bin/env python3
"""Create a source-bound install receipt for an extracted Hermes runtime."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
from pathlib import Path

from hermes_cli.runtime_identity import _installed_content_digest, capture_source_identity


_OID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _artifact_identity(artifact_bytes: bytes) -> dict:
    entries: list[dict] = []
    seen: set[str] = set()
    with tarfile.open(fileobj=io.BytesIO(artifact_bytes), mode="r:gz") as archive:
        try:
            manifest_member = archive.getmember("hermes-runtime-manifest.json")
            manifest_file = archive.extractfile(manifest_member)
            if manifest_file is None:
                raise RuntimeError("runtime artifact manifest is unreadable")
            manifest = json.load(manifest_file)
        except (KeyError, ValueError, tarfile.TarError) as exc:
            raise RuntimeError("runtime artifact manifest is unreadable") from exc
        for member in archive.getmembers():
            if (member.isfile() or member.isdir()) and member.mode & 0o222:
                raise RuntimeError("runtime artifact contains writable content")
            if member.name == "hermes-runtime-manifest.json" or member.isdir():
                continue
            candidate = Path(member.name)
            if candidate.is_absolute() or ".." in candidate.parts or member.name in seen:
                raise RuntimeError("runtime artifact contains an unsafe or duplicate path")
            seen.add(member.name)
            if member.isfile():
                payload_file = archive.extractfile(member)
                if payload_file is None:
                    raise RuntimeError("runtime artifact content is unreadable")
                content = payload_file.read()
                kind = "file"
                executable = bool(member.mode & 0o111)
            elif member.issym():
                content = member.linkname.encode("utf-8")
                kind = "symlink"
                executable = False
            else:
                raise RuntimeError("runtime artifact contains an unsupported entry")
            entries.append({
                "path": member.name,
                "kind": kind,
                "executable": executable,
                "sha256": hashlib.sha256(content).hexdigest(),
            })
    if not isinstance(manifest, dict):
        raise RuntimeError("runtime artifact manifest is invalid")
    source = manifest.get("source")
    artifact_contract = manifest.get("artifact")
    encoded = json.dumps(
        sorted(entries, key=lambda item: item["path"].encode("utf-8")),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    content_sha256 = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
    if (
        manifest.get("contract_version") != 1
        or not isinstance(source, dict)
        or not isinstance(artifact_contract, dict)
        or not isinstance(source.get("revision"), str)
        or not _OID.fullmatch(source["revision"])
        or not isinstance(source.get("tree"), str)
        or not _OID.fullmatch(source["tree"])
        or source.get("dirty") is not False
        or artifact_contract.get("content_sha256") != content_sha256
        or artifact_contract.get("content_entry_count") != len(entries)
        or not _SHA256.fullmatch(content_sha256)
    ):
        raise RuntimeError("runtime artifact content does not match its manifest")
    return {
        "source_revision": source["revision"],
        "source_tree": source["tree"],
        "source_content_sha256": content_sha256,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def create_receipt(
    artifact: Path,
    runtime_root: Path,
    agent_root: Path,
    python_runtime_root: Path,
    python_bin: Path,
    output: Path,
) -> dict:
    artifact = artifact.resolve(strict=True)
    runtime_root = runtime_root.resolve(strict=True)
    agent_root = agent_root.resolve(strict=True)
    if os.pathsep in str(agent_root):
        raise RuntimeError("Hermes source path cannot contain the Python path delimiter")
    python_runtime_root = python_runtime_root.resolve(strict=True)
    python_bin = Path(os.path.abspath(python_bin))
    python_bin = python_bin.parent.resolve(strict=True) / python_bin.name
    if not python_bin.is_symlink() or not python_bin.is_file() or not os.access(python_bin, os.X_OK):
        raise RuntimeError("Hermes Python entry point must be the virtualenv Python symlink")
    python_realpath = python_bin.resolve(strict=True)
    target_metadata = python_realpath.stat()
    if (
        not python_realpath.is_file()
        or python_runtime_root not in python_realpath.parents
        or target_metadata.st_mode & 0o222
    ):
        raise RuntimeError("Hermes Python target must be inside the frozen Python runtime")
    if python_bin.parent.name != "bin":
        raise RuntimeError("Hermes Python entry point must be the virtualenv bin/python symlink")
    environment_root = python_bin.parent.parent.resolve(strict=True)
    if python_bin != environment_root / "bin" / "python":
        raise RuntimeError("Hermes Python entry path must use the canonical runtime environment")
    if (
        agent_root.parent != runtime_root
        or environment_root.parent != runtime_root
        or python_runtime_root.parent != runtime_root
        or len({agent_root, environment_root, python_runtime_root}) != 3
    ):
        raise RuntimeError(
            "Hermes source, Python runtime, and environment must be distinct runtime-root children"
        )
    if runtime_root.stat().st_mode & 0o222:
        raise RuntimeError("Hermes runtime root must be non-writable")
    environment_identity = _installed_content_digest(
        environment_root, allowed_external_python=python_realpath
    )
    python_runtime_identity = _installed_content_digest(python_runtime_root)
    if environment_identity is None:
        raise RuntimeError("Hermes Python environment must be non-writable and content-addressable")
    if python_runtime_identity is None:
        raise RuntimeError("Hermes Python runtime must be non-writable and content-addressable")
    output = output.resolve()
    if output == runtime_root or runtime_root in output.parents:
        raise RuntimeError("runtime receipt must live outside the immutable runtime root")
    artifact_bytes = artifact.read_bytes()
    artifact_identity = _artifact_identity(artifact_bytes)
    identity = capture_source_identity(agent_root, python_bin)
    if identity.get("source_identity_provenance") != "embedded-manifest":
        raise RuntimeError("installed Hermes content does not match its embedded manifest")
    for key in ("source_revision", "source_tree", "source_content_sha256"):
        if identity.get(key) != artifact_identity[key]:
            raise RuntimeError("runtime artifact and installed Hermes content do not match")
    receipt = {
        "contractVersion": 1,
        "runtimeRoot": str(runtime_root),
        "agentRoot": str(agent_root),
        "pythonBin": str(python_bin),
        "pythonRealpath": str(python_realpath),
        "pythonEntrySha256": sha256(python_realpath),
        "pythonRuntimeRoot": str(python_runtime_root),
        "pythonRuntimeContentSha256": python_runtime_identity[0],
        "pythonRuntimeEntryCount": python_runtime_identity[1],
        "pythonEnvironmentRoot": str(environment_root),
        "pythonEnvironmentContentSha256": environment_identity[0],
        "pythonEnvironmentEntryCount": environment_identity[1],
        "sourceRevision": identity["source_revision"],
        "sourceTree": identity["source_tree"],
        "sourceContentSha256": identity["source_content_sha256"],
        "artifactSha256": f"sha256:{hashlib.sha256(artifact_bytes).hexdigest()}",
    }
    encoded = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
    try:
        temporary.chmod(0o400)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {"receipt": receipt, "receiptSha256": f"sha256:{hashlib.sha256(encoded).hexdigest()}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--agent-root", type=Path, required=True)
    parser.add_argument("--python-runtime-root", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = create_receipt(
            args.artifact,
            args.runtime_root,
            args.agent_root,
            args.python_runtime_root,
            args.python_bin,
            args.output,
        )
    except (OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
