"""Secret-free effective runtime identity for supported control clients."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import socket
import stat
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urlsplit

from hermes_cli.broker_mode import (
    QUALIFICATION_MAX_OUTPUT_TOKENS,
    QUALIFICATION_SEED,
    QUALIFICATION_TEMPERATURE,
    broker_mode_requested,
    qualification_mode_requested,
)


RUNTIME_IDENTITY_CONTRACT_VERSION = 1
_PACKAGED_MANIFEST = "hermes-runtime-manifest.json"
_OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_PREFIXED_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_LAUNCHER_SOURCE_SHA256_ENV = "HERMES_RUNTIME_VERIFIED_SOURCE_SHA256"
_LAUNCHER_SOURCE_FINGERPRINT_ENV = "HERMES_RUNTIME_VERIFIED_SOURCE_FINGERPRINT"
_TAILSCALE_V4 = ipaddress.ip_network("100.64.0.0/10")
_TAILSCALE_V6 = ipaddress.ip_network("fd7a:115c:a1e0::/48")
_LOCAL_SUFFIXES = (".local", ".lan", ".home", ".internal")
_Resolver = Callable[[str], Iterable[str]]


def _legacy_ip_literal(hostname: str) -> ipaddress.IPv4Address | None:
    """Parse IPv4 forms accepted by operating-system HTTP stacks."""

    if not re.fullmatch(r"(?:0[xX][0-9a-fA-F]+|[0-9.]+)", hostname):
        return None
    try:
        return ipaddress.IPv4Address(socket.inet_aton(hostname))
    except OSError:
        return None


def _ip_literal(hostname: str):
    candidate = hostname.split("%", 1)[0]
    try:
        return ipaddress.ip_address(candidate)
    except ValueError:
        return _legacy_ip_literal(candidate)


@lru_cache(maxsize=64)
def _resolve_host_addresses(hostname: str) -> tuple[str, ...]:
    try:
        rows = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except OSError:
        return ()
    return tuple(sorted({str(row[4][0]).split("%", 1)[0] for row in rows}))


def _address_class(address) -> str:
    if address.is_loopback:
        return "loopback"
    if address in _TAILSCALE_V4 or address in _TAILSCALE_V6:
        return "tailnet"
    if address.is_link_local:
        return "link-local"
    if address.is_private:
        return "private-network"
    return "public"


def _classify_host(hostname: str, resolver: _Resolver) -> str:
    if hostname == "localhost" or hostname.endswith(".localhost"):
        return "loopback"
    literal = _ip_literal(hostname)
    if literal is not None:
        return _address_class(literal)
    resolved = [_ip_literal(value) for value in resolver(hostname)]
    classes = {_address_class(value) for value in resolved if value is not None}
    for candidate in ("loopback", "tailnet", "link-local", "private-network"):
        if candidate in classes:
            return candidate
    if hostname.endswith(".ts.net"):
        return "tailnet"
    if hostname.endswith(_LOCAL_SUFFIXES):
        return "local-network"
    return "public"


def endpoint_identity(
    base_url: object,
    *,
    resolver: _Resolver = _resolve_host_addresses,
) -> dict:
    """Return a secret-free endpoint class and normalized-origin fingerprint."""

    value = str(base_url or "").strip()
    if not value:
        return {
            "base_url_class": "provider-default",
            "base_url_fingerprint": None,
        }
    try:
        parsed = urlsplit(value)
        scheme = parsed.scheme.lower()
        hostname = (parsed.hostname or "").lower().rstrip(".")
        port = parsed.port
    except (TypeError, ValueError):
        return {
            "base_url_class": "invalid",
            "base_url_fingerprint": None,
        }
    if scheme not in {"http", "https"} or not hostname:
        return {
            "base_url_class": "invalid",
            "base_url_fingerprint": None,
        }
    try:
        hostname = hostname.encode("idna").decode("ascii")
    except UnicodeError:
        return {
            "base_url_class": "invalid",
            "base_url_fingerprint": None,
        }
    literal = _ip_literal(hostname)
    normalized_host = str(literal) if literal is not None else hostname
    display_host = f"[{normalized_host}]" if ":" in normalized_host else normalized_host
    default_port = 80 if scheme == "http" else 443
    port_suffix = "" if port in {None, default_port} else f":{port}"
    normalized_origin = f"{scheme}://{display_host}{port_suffix}"
    fingerprint = hashlib.sha256(normalized_origin.encode("utf-8")).hexdigest()
    return {
        "base_url_class": _classify_host(hostname, resolver),
        "base_url_fingerprint": f"sha256:{fingerprint}",
    }


def classify_base_url(base_url: object, *, resolver: _Resolver = _resolve_host_addresses) -> str:
    """Classify an endpoint without returning its URL, host, port, or path."""

    return str(endpoint_identity(base_url, resolver=resolver)["base_url_class"])


def _run_git(source_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(source_root), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_pth_content(path: Path, root: Path, content: bytes) -> bool:
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return False
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("import ", "import\t")):
            return False
        candidate = Path(os.path.abspath(path.parent / line))
        if candidate != root and root not in candidate.parents:
            return False
        if candidate.exists():
            try:
                resolved = candidate.resolve(strict=True)
            except (OSError, RuntimeError):
                return False
            if resolved != root and root not in resolved.parents:
                return False
    return True


def _installed_content_digest(
    root: Path,
    *,
    allowed_external_python: Path | None = None,
) -> tuple[str, int] | None:
    entries: list[dict] = []
    try:
        if root.stat().st_mode & 0o222:
            return None
        for directory, names, files in os.walk(root, followlinks=False):
            directory_path = Path(directory)
            names.sort()
            files.sort()
            for name in [*names, *files]:
                candidate = directory_path / name
                relative = candidate.relative_to(root).as_posix()
                if relative == _PACKAGED_MANIFEST:
                    continue
                metadata = candidate.lstat()
                if candidate.is_symlink():
                    resolved = candidate.resolve(strict=True)
                    inside_root = resolved == root or root in resolved.parents
                    python_link = relative in {"bin/python", "bin/python3", "bin/python3.11"}
                    if not inside_root and not (
                        allowed_external_python is not None
                        and python_link
                        and resolved == allowed_external_python
                    ):
                        return None
                    entries.append({
                        "path": relative,
                        "kind": "symlink",
                        "executable": False,
                        "sha256": hashlib.sha256(
                            os.readlink(candidate).encode("utf-8")
                        ).hexdigest(),
                    })
                    if name in names:
                        names.remove(name)
                elif candidate.is_dir():
                    if metadata.st_mode & 0o222:
                        return None
                    continue
                elif candidate.is_file():
                    if metadata.st_mode & 0o222:
                        return None
                    content = candidate.read_bytes()
                    if candidate.suffix == ".pth" and not _safe_pth_content(
                        candidate, root, content
                    ):
                        return None
                    entries.append({
                        "path": relative,
                        "kind": "file",
                        "executable": bool(metadata.st_mode & 0o111),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    })
                else:
                    return None
    except (OSError, RuntimeError, ValueError):
        return None
    encoded = json.dumps(
        sorted(entries, key=lambda item: item["path"].encode("utf-8")),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}", len(entries)


def _installed_stat_fingerprint(root: Path) -> str | None:
    """Fingerprint every entry's metadata under root without reading contents.

    Byte for byte the format of Maestro's installedStatFingerprint for one
    root: path, mode, size, inode, and modification and change times in
    nanoseconds, depth first in UTF-16 name order. A write, chmod, rename or
    replacement moves the change time or the inode, and a process cannot set
    the change time back.
    """

    digest = hashlib.sha256()

    def line(text: str) -> None:
        digest.update(text.encode("utf-8", "surrogateescape"))

    def walk(directory: Path, prefix: str) -> None:
        names = sorted(
            os.listdir(directory),
            key=lambda name: name.encode("utf-16-be", "surrogatepass"),
        )
        for name in names:
            candidate = directory / name
            relative = f"{prefix}{name}"
            metadata = candidate.lstat()
            line(
                f"{relative}\0{metadata.st_mode}\0{metadata.st_size}\0{metadata.st_ino}"
                f"\0{metadata.st_mtime_ns}\0{metadata.st_ctime_ns}\n"
            )
            if stat.S_ISDIR(metadata.st_mode):
                walk(candidate, f"{relative}/")

    try:
        metadata = root.lstat()
        line(
            f"{root}\0{metadata.st_mode}\0{metadata.st_ino}"
            f"\0{metadata.st_mtime_ns}\0{metadata.st_ctime_ns}\n"
        )
        walk(root, "")
    except (OSError, RuntimeError, UnicodeError, RecursionError):
        return None
    return f"sha256:{digest.hexdigest()}"


def _launcher_verified_content(root: Path, content_sha256: str) -> bool:
    """Whether the launcher already hashed this exact tree to content_sha256.

    Maestro checks the pinned runtime's contents against its receipt before
    it starts a gateway, then passes the digest it verified and the metadata
    fingerprint of the tree at that moment. Matching metadata means the
    contents are the ones it hashed, so the gateway can skip its own walk
    (0.4 s warm, 1.7 s cold on Scout). Anything else hashes the contents.
    """

    verified = os.environ.get(_LAUNCHER_SOURCE_SHA256_ENV, "").strip()
    fingerprint = os.environ.get(_LAUNCHER_SOURCE_FINGERPRINT_ENV, "").strip()
    if verified != content_sha256 or not _PREFIXED_DIGEST.fullmatch(fingerprint):
        # branch-divergence: intentional — no launcher claim, so the caller hashes the contents itself
        return False
    return _installed_stat_fingerprint(root) == fingerprint


def _read_packaged_source(manifest_path: Path) -> dict | None:
    try:
        if manifest_path.stat().st_mode & 0o222:
            return None
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(raw, dict) or raw.get("contract_version") != 1:
        return None
    artifact = raw.get("artifact")
    source = raw.get("source")
    locks = raw.get("locks")
    if (
        not isinstance(artifact, dict)
        or artifact.get("format") != "tar.gz"
        or artifact.get("filesystem") != "read-only"
        or not isinstance(artifact.get("content_sha256"), str)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", artifact["content_sha256"])
        or not isinstance(artifact.get("content_entry_count"), int)
        or artifact["content_entry_count"] < 1
        or not isinstance(source, dict)
        or not isinstance(locks, dict)
        or not locks
    ):
        return None
    revision = source.get("revision")
    tree = source.get("tree")
    if (
        not isinstance(revision, str)
        or not _OBJECT_ID.fullmatch(revision)
        or not isinstance(tree, str)
        or not _OBJECT_ID.fullmatch(tree)
        or source.get("dirty") is not False
    ):
        return None
    for name, expected in locks.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(expected, str)
            or not _DIGEST.fullmatch(expected)
        ):
            return None
        lock_path = manifest_path.parent / name
        try:
            if _sha256(lock_path) != expected:
                return None
        except OSError:
            return None
    expected_content = (artifact["content_sha256"], artifact["content_entry_count"])
    if _launcher_verified_content(manifest_path.parent, artifact["content_sha256"]):
        verified_by = "launcher"
    elif _installed_content_digest(manifest_path.parent) == expected_content:
        verified_by = "hermes"
    else:
        return None
    return {
        "source_revision": revision,
        "source_tree": tree,
        "source_dirty": False,
        "source_identity_provenance": "embedded-manifest",
        "source_content_sha256": artifact["content_sha256"],
        "source_content_verified_by": verified_by,
    }


def capture_source_identity(source_root: Path, python_executable: Path) -> dict:
    """Capture source identity once, rejecting Git metadata from a parent repo."""

    source_root = source_root.resolve()
    base = {
        "python_realpath": str(python_executable.resolve()),
        "source_realpath": str(source_root),
    }
    manifest_path = source_root / _PACKAGED_MANIFEST
    if manifest_path.exists():
        packaged = _read_packaged_source(manifest_path)
        if packaged is None:
            return {
                **base,
                "source_revision": None,
                "source_tree": None,
                "source_dirty": None,
                "source_identity_provenance": "invalid-embedded-manifest",
                "source_content_sha256": None,
            }
        return {**base, **packaged}

    top_level = _run_git(source_root, "rev-parse", "--show-toplevel")
    if top_level is None or Path(top_level).resolve() != source_root:
        return {
            **base,
            "source_revision": None,
            "source_tree": None,
            "source_dirty": None,
            "source_identity_provenance": "unverified",
            "source_content_sha256": None,
        }
    revision = _run_git(source_root, "rev-parse", "HEAD^{commit}")
    tree = _run_git(source_root, "rev-parse", f"{revision}^{{tree}}") if revision else None
    try:
        diff = subprocess.run(
            ["git", "-C", str(source_root), "diff", "--quiet", revision or "", "--"],
            check=False,
            capture_output=True,
            timeout=2,
        )
        untracked = _run_git(source_root, "ls-files", "--others", "--exclude-standard")
        dirty = None if diff.returncode not in {0, 1} or untracked is None else diff.returncode == 1 or bool(untracked)
    except (OSError, subprocess.SubprocessError):
        dirty = None
    if revision is None or tree is None or dirty is None:
        return {
            **base,
            "source_revision": None,
            "source_tree": None,
            "source_dirty": None,
            "source_identity_provenance": "unverified",
            "source_content_sha256": None,
        }
    return {
        **base,
        "source_revision": revision,
        "source_tree": tree,
        "source_dirty": dirty,
        "source_identity_provenance": "git",
        "source_content_sha256": None,
    }


_BOOTSTRAP_SOURCE_IDENTITY = capture_source_identity(
    Path(__file__).resolve().parents[1],
    Path(sys.executable),
)


def source_identity() -> dict:
    """Return the source identity captured during module bootstrap."""

    return dict(_BOOTSTRAP_SOURCE_IDENTITY)


def effective_runtime_identity(agent) -> dict:
    """Build the versioned, secret-free runtime contract for a built agent."""

    qualification = qualification_mode_requested()
    broker_mode = broker_mode_requested()
    broker = {
        "contract_version": 1,
        "mode": broker_mode,
        "available": False,
    }
    if broker_mode:
        try:
            from hermes_cli.broker_mode import (
                BROKER_CHANNEL_FD,
                BROKER_PROTOCOL_VERSION,
                broker_route_id,
            )

            os.fstat(BROKER_CHANNEL_FD)
            broker.update(
                available=True,
                protocol_version=BROKER_PROTOCOL_VERSION,
                route_id=broker_route_id(),
                transport="inherited-fd",
                supported_auxiliary_roles=["vision"],
            )
            from hermes_constants import get_hermes_home

            broker["profile_home_realpath"] = str(get_hermes_home().resolve())
        except (OSError, RuntimeError):
            pass
    return {
        "contract_version": RUNTIME_IDENTITY_CONTRACT_VERSION,
        "provider": str(getattr(agent, "provider", "") or ""),
        "model": str(getattr(agent, "model", "") or ""),
        **endpoint_identity(getattr(agent, "base_url", None)),
        **source_identity(),
        "broker": broker,
        "qualification": {
            "contract_version": 1,
            "mode": qualification,
            "tool_free": qualification,
            "tool_surface": "empty" if qualification else "standard",
            "tool_execution": "rejected" if qualification else "standard",
            "sampling": {
                "temperature": QUALIFICATION_TEMPERATURE if qualification else None,
                "seed": QUALIFICATION_SEED if qualification else None,
                "max_output_tokens": (
                    QUALIFICATION_MAX_OUTPUT_TOKENS if qualification else None
                ),
            },
        },
    }
