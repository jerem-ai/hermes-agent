from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
import pytest

from hermes_cli.broker_transport import BrokerOpenAIClient, BrokerProtocolError


def _serve_once(peer: socket.socket, handler):
    def run():
        stream = peer.makefile("rwb", buffering=0)
        request = json.loads(stream.readline())
        for event in handler(request):
            stream.write(json.dumps(event, ensure_ascii=False).encode() + b"\n")
        stream.close()
        peer.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def test_broker_client_accepts_unicode_and_keeps_credentials_outside_request(monkeypatch):
    gateway, broker = socket.socketpair()
    gateway.set_inheritable(True)
    monkeypatch.setenv("HERMES_BROKER_ROUTE_ID", "route/研究-Ω")
    captured = {}

    def handler(request):
        captured.update(request)
        yield {
            "v": 1,
            "type": "response",
            "id": request["id"],
            "body": {
                "id": "mock-1",
                "choices": [{"message": {"role": "assistant", "content": "γειά σου 世界"}}],
                "usage": {"prompt_tokens": 7, "completion_tokens": 4},
            },
        }

    thread = _serve_once(broker, handler)
    client = BrokerOpenAIClient("模型/α", fd=gateway.fileno())
    response = client.chat.completions.create(
        model="模型/α",
        messages=[{"role": "user", "content": "hello"}],
    )
    thread.join(timeout=2)
    gateway.close()

    assert response.choices[0].message.content == "γειά σου 世界"
    assert captured["route"] == "route/研究-Ω"
    assert captured["body"]["model"] == "模型/α"
    assert "api_key" not in json.dumps(captured).lower()
    assert "authorization" not in json.dumps(captured).lower()


def test_broker_client_streams_correlated_chunks(monkeypatch):
    gateway, broker = socket.socketpair()
    monkeypatch.setenv("HERMES_BROKER_ROUTE_ID", "local_primary")

    def handler(request):
        for text in ("one", " two"):
            yield {
                "v": 1,
                "type": "chunk",
                "id": request["id"],
                "body": {"choices": [{"delta": {"content": text}, "finish_reason": None}]},
            }
        yield {"v": 1, "type": "done", "id": request["id"]}

    thread = _serve_once(broker, handler)
    client = BrokerOpenAIClient("qwen", fd=gateway.fileno())
    chunks = list(client.chat.completions.create(
        model="qwen", messages=[], stream=True,
    ))
    thread.join(timeout=2)
    gateway.close()

    assert "".join(chunk.choices[0].delta.content for chunk in chunks) == "one two"


def test_broker_clients_serialize_exchanges_on_one_inherited_descriptor(monkeypatch):
    gateway, broker = socket.socketpair()
    monkeypatch.setenv("HERMES_BROKER_ROUTE_ID", "route/研究-Ω")
    first_seen = threading.Event()
    received_before_first_response = []

    def serve():
        stream = broker.makefile("rwb", buffering=0)
        first = json.loads(stream.readline())
        first_seen.set()
        time.sleep(0.1)
        ready, _, _ = select.select([broker], [], [], 0)
        received_before_first_response.append(bool(ready))
        stream.write(json.dumps({
            "v": 1, "type": "response", "id": first["id"],
            "body": {"choices": [{"message": {"content": "first 研究"}}]},
        }).encode() + b"\n")
        second = json.loads(stream.readline())
        stream.write(json.dumps({
            "v": 1, "type": "response", "id": second["id"],
            "body": {"choices": [{"message": {"content": "second Ω"}}]},
        }).encode() + b"\n")
        stream.close()
        broker.close()

    import select

    server = threading.Thread(target=serve, daemon=True)
    server.start()
    first_client = BrokerOpenAIClient("model/研究", fd=gateway.fileno())
    second_client = BrokerOpenAIClient("model/研究", fd=gateway.fileno())
    responses = []

    def call(client):
        response = client.chat.completions.create(model="model/研究", messages=[])
        responses.append(response.choices[0].message.content)

    first_thread = threading.Thread(target=call, args=(first_client,))
    second_thread = threading.Thread(target=call, args=(second_client,))
    first_thread.start()
    assert first_seen.wait(timeout=2)
    second_thread.start()
    first_thread.join(timeout=2)
    second_thread.join(timeout=2)
    server.join(timeout=2)
    gateway.close()

    assert received_before_first_response == [False]
    assert responses == ["first 研究", "second Ω"]


def test_broker_descriptor_is_not_inherited_by_tool_children(monkeypatch):
    gateway, broker = socket.socketpair()
    gateway.set_inheritable(True)
    monkeypatch.setenv("HERMES_BROKER_ROUTE_ID", "local_primary")
    client = BrokerOpenAIClient("qwen", fd=gateway.fileno())

    probe = subprocess.run(
        [sys.executable, "-c", f"import os; os.fstat({gateway.fileno()})"],
        text=True,
        capture_output=True,
        check=False,
    )
    gateway.close()
    broker.close()

    assert client.route_id == "local_primary"
    assert probe.returncode != 0
    assert "Bad file descriptor" in probe.stderr


def test_broker_client_rejects_model_escape_and_wrong_correlation(monkeypatch):
    gateway, broker = socket.socketpair()
    monkeypatch.setenv("HERMES_BROKER_ROUTE_ID", "local_primary")
    client = BrokerOpenAIClient("qwen", fd=gateway.fileno())
    with pytest.raises(BrokerProtocolError, match="bound model"):
        client.chat.completions.create(model="other", messages=[])

    def handler(request):
        yield {"v": 1, "type": "response", "id": "wrong", "body": {}}

    thread = _serve_once(broker, handler)
    with pytest.raises(BrokerProtocolError, match="correlation"):
        client.chat.completions.create(model="qwen", messages=[])
    thread.join(timeout=2)
    gateway.close()


def test_broker_client_fails_closed_without_descriptor(monkeypatch):
    monkeypatch.setenv("HERMES_BROKER_ROUTE_ID", "local_primary")
    with pytest.raises(BrokerProtocolError, match="unavailable"):
        BrokerOpenAIClient("qwen", fd=987)


def test_broker_client_omits_sdk_only_values_and_rejects_unknown_objects(monkeypatch):
    gateway, broker = socket.socketpair()
    monkeypatch.setenv("HERMES_BROKER_ROUTE_ID", "local_primary")
    captured = {}

    def handler(request):
        captured.update(request)
        yield {
            "v": 1,
            "type": "response",
            "id": request["id"],
            "body": {"choices": [{"message": {"content": "ok"}}]},
        }

    thread = _serve_once(broker, handler)
    client = BrokerOpenAIClient("qwen", fd=gateway.fileno())
    client.chat.completions.create(
        model="qwen",
        messages=[],
        timeout=object(),
        extra_headers={"Authorization": "must-not-cross"},
        extra_body={"seed": 7},
    )
    thread.join(timeout=2)
    assert captured["body"] == {"model": "qwen", "messages": [], "seed": 7}

    with pytest.raises(BrokerProtocolError, match="unsupported value type"):
        client.chat.completions.create(model="qwen", messages=[], metadata=object())
    gateway.close()


def test_broker_mode_never_imports_user_plugin_code(monkeypatch, tmp_path: Path):
    home = tmp_path / "hermes-home"
    plugin = home / "plugins" / "hostile"
    marker = tmp_path / "plugin-imported"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        "name: hostile\nversion: 1.0.0\n",
        encoding="utf-8",
    )
    (plugin / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n"
        "def register(ctx):\n    raise RuntimeError('must not run')\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("_HERMES_INTERNAL_BROKER_MODE", "1")
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli.plugins import discover_plugins, invoke_hook

    discover_plugins(force=True)
    assert invoke_hook("pre_tool_call", name="shell") == []
    assert not marker.exists()
