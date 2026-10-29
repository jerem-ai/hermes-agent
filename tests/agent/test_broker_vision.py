import json
import socket
import threading
from pathlib import Path

import pytest

from agent import auxiliary_client as auxiliary
from hermes_cli import broker_transport
from hermes_cli.broker_transport import BrokerOpenAIClient, BrokerProtocolError


def prepare(monkeypatch, *, error=None):
    home = Path(__import__('os').environ['HERMES_HOME'])
    (home / 'config.yaml').write_text('auxiliary:\n  free_only: true\n  vision:\n    provider: broker\n    model: vision/研究\n    fallback_chain: []\n')
    monkeypatch.setenv('_HERMES_INTERNAL_BROKER_MODE', '1')
    monkeypatch.setenv('HERMES_BROKER_ROUTE_ID', 'opaque/run-Ω')
    gateway, peer = socket.socketpair()
    captured = []
    # Select a test descriptor while retaining the actual transport and shared lock.
    monkeypatch.setattr(broker_transport, 'build_broker_openai_client',
                        lambda model: BrokerOpenAIClient(model, fd=gateway.fileno()))

    def serve():
        with peer, peer.makefile('rwb', buffering=0) as stream:
            request = json.loads(stream.readline())
            captured.append(request)
            event = {'v': 1, 'id': request['id']}
            if error:
                event.update(type='error', code=error)
            else:
                event.update(type='response', body={'model': 'vision/研究', 'choices': [{'message': {'content': 'Zoë / 李'}}], 'usage': {'prompt_tokens': 7, 'completion_tokens': 4}})
            stream.write(json.dumps(event).encode() + b'\n')

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    monkeypatch.setattr(socket.socket, 'connect', lambda *a, **k: (_ for _ in ()).throw(AssertionError('broker vision attempted a network connection')))
    return gateway, thread, captured


@pytest.mark.asyncio
async def test_vision_config_uses_inherited_channel_with_complete_unicode_image(monkeypatch):
    gateway, thread, captured = prepare(monkeypatch)
    image = 'data:image/png;base64,' + 'a' * (512 * 1024)
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'Read Zoë / 李'}, {'type': 'image_url', 'image_url': {'url': image}}]}]
    try:
        response = await auxiliary.async_call_llm(task='vision', messages=messages, max_tokens=32, temperature=0)
        assert response.choices[0].message.content == 'Zoë / 李'
        thread.join(timeout=2)
        assert captured[0]['route'] == 'opaque/run-Ω'
        assert captured[0]['body']['messages'] == messages
        assert captured[0]['body']['model'] == 'vision/研究'
        assert captured[0]['body'].get('stream', False) is False
    finally:
        gateway.close()


@pytest.mark.asyncio
async def test_broker_refusal_does_not_fall_back_to_a_network_provider(monkeypatch):
    gateway, thread, captured = prepare(monkeypatch, error='broker_vision_limit')
    try:
        with pytest.raises(BrokerProtocolError, match='broker_vision_limit'):
            await auxiliary.async_call_llm(task='vision', messages=[{'role': 'user', 'content': 'inspect'}])
        thread.join(timeout=2)
        assert len(captured) == 1
    finally:
        gateway.close()


def test_broker_provider_without_process_capability_refuses_before_network(monkeypatch):
    monkeypatch.delenv('_HERMES_INTERNAL_BROKER_MODE', raising=False)
    monkeypatch.setattr(socket.socket, 'connect', lambda *a, **k: pytest.fail('unexpected network connection'))
    with pytest.raises(RuntimeError, match='inherited capability'):
        auxiliary.resolve_vision_provider_client(provider='broker', model='vision/研究')


def test_runtime_advertises_vision_only_with_a_live_inherited_channel(monkeypatch):
    from types import SimpleNamespace
    from hermes_cli import broker_mode, runtime_identity
    gateway, peer = socket.socketpair()
    monkeypatch.setenv('_HERMES_INTERNAL_BROKER_MODE', '1')
    monkeypatch.setenv('HERMES_BROKER_ROUTE_ID', 'role-capability')
    monkeypatch.setattr(broker_mode, 'BROKER_CHANNEL_FD', gateway.fileno())
    try:
        identity = runtime_identity.effective_runtime_identity(SimpleNamespace(provider='broker', model='text-pin'))
        assert identity['broker']['available'] is True
        assert 'vision' in identity['broker']['supported_auxiliary_roles']
        gateway.close()
        identity = runtime_identity.effective_runtime_identity(SimpleNamespace(provider='broker', model='text-pin'))
        assert identity['broker']['available'] is False
        assert 'supported_auxiliary_roles' not in identity['broker']
    finally:
        gateway.close()
        peer.close()
