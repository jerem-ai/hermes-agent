"""The runner's status glyphs must not crash narrow console encodings.

On native Windows, piped or legacy-console stdio defaults to cp1252, which
cannot encode the runner's ✓/✗ progress glyphs — before the fix, the first
per-file status line killed the whole run with UnicodeEncodeError. The
failure depends only on the stream's encoding, so these tests pin it on
every OS by building a cp1252 stream explicitly.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNNER_PATH = REPO_ROOT / "scripts" / "run_tests_parallel.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_tests_parallel", _RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cp1252_stream() -> tuple[io.TextIOWrapper, io.BytesIO]:
    raw = io.BytesIO()
    return io.TextIOWrapper(raw, encoding="cp1252", errors="strict"), raw


def test_glyph_safe_stdio_survives_cp1252(monkeypatch) -> None:
    mod = _load_runner()
    stream, raw = _cp1252_stream()
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", stream)

    mod._make_stdio_glyph_safe()
    print("✓ tests/foo.py (3 tests, 1.2s) ✗")
    sys.stdout.flush()

    out = raw.getvalue()
    assert "✓".encode("utf-8") in out, "stream should now carry UTF-8 glyphs"
    assert b"tests/foo.py (3 tests, 1.2s)" in out, "line content must survive"


def test_glyph_safe_stdio_noop_without_reconfigure(monkeypatch) -> None:
    # Streams without .reconfigure (e.g. pytest's capture buffers, plain
    # StringIO) must pass through untouched instead of raising.
    mod = _load_runner()
    plain = io.StringIO()
    monkeypatch.setattr(sys, "stdout", plain)
    monkeypatch.setattr(sys, "stderr", plain)

    mod._make_stdio_glyph_safe()
    print("✓ still fine")

    assert "✓ still fine" in plain.getvalue()


def test_open_file_limit_raises_low_soft_ceiling(monkeypatch) -> None:
    mod = _load_runner()

    class FakeResource:
        RLIMIT_NOFILE = 7
        RLIM_INFINITY = -1

        def __init__(self):
            self.limit = (256, 8192)

        def getrlimit(self, _kind):
            return self.limit

        def setrlimit(self, _kind, value):
            self.limit = value

    fake = FakeResource()
    monkeypatch.setitem(sys.modules, "resource", fake)

    assert mod._raise_open_file_limit(4096) is True
    assert fake.limit == (4096, 8192)


def test_open_file_limit_preserves_adequate_soft_ceiling(monkeypatch) -> None:
    mod = _load_runner()

    class FakeResource:
        RLIMIT_NOFILE = 7
        RLIM_INFINITY = -1

        @staticmethod
        def getrlimit(_kind):
            return (8192, 8192)

        @staticmethod
        def setrlimit(_kind, _value):
            raise AssertionError("an adequate limit must remain untouched")

    monkeypatch.setitem(sys.modules, "resource", FakeResource())

    assert mod._raise_open_file_limit(4096) is True
