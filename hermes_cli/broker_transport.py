"""OpenAI-compatible client facade over Maestro's run-scoped broker channel.

The gateway receives one connected descriptor at process launch. It never
receives a provider credential or network endpoint. The descriptor is marked
non-inheritable before any agent or tool subprocess can start.
"""

from __future__ import annotations

import json
import math
import os
import threading
import uuid
from types import SimpleNamespace
from typing import Any, Iterator

from hermes_cli.broker_mode import (
    BROKER_CHANNEL_FD,
    BROKER_PROTOCOL_VERSION,
    broker_route_id,
)


class BrokerProtocolError(RuntimeError):
    pass


def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


_OMIT = object()
_CLIENT_ONLY_KEYS = {"extra_headers", "extra_query", "timeout"}


def _wire_value(value: Any) -> Any:
    """Convert SDK request values to the provider-neutral JSON wire shape."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise BrokerProtocolError("Broker request contains a non-finite number")
        return value
    if value.__class__.__name__ in {"NotGiven", "Omit"}:
        return _OMIT
    if isinstance(value, (list, tuple)):
        converted = []
        for item in value:
            wire = _wire_value(item)
            if wire is not _OMIT:
                converted.append(wire)
        return converted
    if isinstance(value, dict):
        converted = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise BrokerProtocolError("Broker request contains a non-string object key")
            wire = _wire_value(item)
            if wire is not _OMIT:
                converted[key] = wire
        return converted
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _wire_value(model_dump(mode="json", exclude_none=True))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _wire_value(to_dict())
    raise BrokerProtocolError(
        f"Broker request contains unsupported value type: {value.__class__.__name__}"
    )


def _request_body(kwargs: dict[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in _CLIENT_ONLY_KEYS:
            continue
        wire = _wire_value(value)
        if wire is not _OMIT:
            body[key] = wire
    extra_body = body.pop("extra_body", None)
    if extra_body is not None:
        if not isinstance(extra_body, dict):
            raise BrokerProtocolError("Broker extra_body must be an object")
        for key, value in extra_body.items():
            if key in body and body[key] != value:
                raise BrokerProtocolError("Broker extra_body conflicts with a bound request field")
            body[key] = value
    return body


class _BrokerChannel:
    def __init__(self, fd: int = BROKER_CHANNEL_FD):
        try:
            os.fstat(fd)
            os.set_inheritable(fd, False)
            self._stream = os.fdopen(fd, "r+b", buffering=0, closefd=False)
        except OSError as exc:
            raise BrokerProtocolError("Broker capability is unavailable") from exc
        self._lock = threading.Lock()
        self._route_id = broker_route_id()

    @property
    def route_id(self) -> str:
        return self._route_id

    def exchange(self, body: dict[str, Any], *, stream: bool) -> Iterator[Any]:
        request_id = uuid.uuid4().hex
        request = {
            "v": BROKER_PROTOCOL_VERSION,
            "type": "request",
            "id": request_id,
            "route": self._route_id,
            "stream": stream,
            "body": body,
        }
        try:
            encoded = json.dumps(request, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise BrokerProtocolError("Broker request is not JSON serializable") from exc
        if len(encoded.encode("utf-8")) > 16 * 1024 * 1024:
            raise BrokerProtocolError("Broker request exceeds the 16 MiB limit")

        with self._lock:
            self._stream.write(encoded.encode("utf-8") + b"\n")
            while True:
                raw = self._stream.readline(16 * 1024 * 1024 + 1)
                if not raw:
                    raise BrokerProtocolError("Broker capability closed before a terminal receipt")
                if len(raw) > 16 * 1024 * 1024:
                    raise BrokerProtocolError("Broker response exceeds the 16 MiB limit")
                try:
                    event = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise BrokerProtocolError("Broker emitted malformed JSON") from exc
                if not isinstance(event, dict) or event.get("v") != BROKER_PROTOCOL_VERSION:
                    raise BrokerProtocolError("Broker protocol version mismatch")
                if event.get("id") != request_id:
                    raise BrokerProtocolError("Broker response correlation mismatch")
                event_type = event.get("type")
                if event_type == "error":
                    code = str(event.get("code") or "broker_request_failed")
                    raise BrokerProtocolError(code)
                if event_type in {"response", "chunk"}:
                    yield _namespace(event.get("body"))
                    if event_type == "response":
                        return
                    continue
                if event_type == "done":
                    return
                raise BrokerProtocolError("Broker emitted an unknown event type")


_CHANNELS_LOCK = threading.Lock()
_CHANNELS: dict[int, tuple[tuple[int, int], _BrokerChannel]] = {}


def _shared_broker_channel(fd: int) -> _BrokerChannel:
    """Return one lock and stream wrapper for each inherited descriptor."""

    try:
        metadata = os.fstat(fd)
    except OSError as exc:
        raise BrokerProtocolError("Broker capability is unavailable") from exc
    identity = (metadata.st_dev, metadata.st_ino)
    with _CHANNELS_LOCK:
        current = _CHANNELS.get(fd)
        if current is None or current[0] != identity:
            current = (identity, _BrokerChannel(fd))
            _CHANNELS[fd] = current
        return current[1]


class _Completions:
    def __init__(self, channel: _BrokerChannel, model: str):
        self._channel = channel
        self._model = model

    def create(self, **kwargs: Any) -> Any:
        body = _request_body(dict(kwargs))
        requested_model = str(body.get("model") or self._model)
        if requested_model != self._model:
            raise BrokerProtocolError("Broker request attempted to change the bound model")
        body["model"] = self._model
        streaming = bool(body.get("stream"))
        results = self._channel.exchange(body, stream=streaming)
        return results if streaming else next(results)


class _Chat:
    def __init__(self, channel: _BrokerChannel, model: str):
        self.completions = _Completions(channel, model)


class BrokerOpenAIClient:
    """The narrow client surface used by Hermes chat-completions paths."""

    def __init__(self, model: str, fd: int = BROKER_CHANNEL_FD):
        if not model or len(model) > 500:
            raise BrokerProtocolError("Broker mode requires a bound model")
        self._channel = _shared_broker_channel(fd)
        self.chat = _Chat(self._channel, model)

    @property
    def route_id(self) -> str:
        return self._channel.route_id

    def close(self) -> None:
        # The process owns the inherited descriptor. Gateway shutdown closes it.
        return None


def build_broker_openai_client(model: str) -> BrokerOpenAIClient:
    return BrokerOpenAIClient(model=model)
