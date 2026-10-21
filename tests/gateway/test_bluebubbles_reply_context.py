"""BlueBubbles replies retain their referent across a fresh session, without cross-chat reads."""
import asyncio
from unittest.mock import AsyncMock

import pytest
import httpx

from gateway.run_inbound import GatewayInboundMixin
from tests.gateway.test_bluebubbles_plugin import _FakeBlueBubblesRequest, _make_adapter


async def _deliver(monkeypatch, parent, *, reference="parent", chat="iMessage;-;fixture", **extra):
    adapter = _make_adapter(monkeypatch, send_read_receipts=False, **extra)
    adapter._api_json = AsyncMock(side_effect=parent if isinstance(parent, BaseException) else None,
                                  return_value={"data": parent})
    handled = []
    adapter.handle_message = AsyncMock(side_effect=handled.append)
    response = await adapter._handle_webhook(_FakeBlueBubblesRequest({
        "type": "new-message", "data": {
            "guid": "new-message", "text": "You didn't do this?",
            "threadOriginatorGuid": reference, "chats": [{"guid": chat}],
            "handle": {"address": "fixture@example.test"},
        },
    }))
    await asyncio.gather(*tuple(adapter._background_tasks))
    assert response.status == 200
    return adapter, handled


@pytest.mark.asyncio
async def test_native_reply_hydrates_long_unicode_quote_into_fresh_turn(monkeypatch):
    quoted = "Committed fixture123.\n" + "工具 evidence @file:/private/fixture.txt\n" * 300
    adapter, handled = await _deliver(monkeypatch, {
        "guid": "parent", "text": quoted, "isFromMe": True,
        "chats": [{"guid": "iMessage;-;fixture"}],
    })
    event = handled[0]
    assert event.reply_to_text == quoted
    assert event.reply_to_is_own_message is True
    prepared = GatewayInboundMixin._prepend_inbound_reply_context(event, event.source, event.text)
    assert quoted in prepared and prepared.endswith("You didn't do this?")
    assert "your previous message" in prepared
    adapter._api_json.assert_awaited_once_with(
        "get", "/api/v1/message/parent?with=chats", timeout=5,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("parent", [
    {"guid": "parent", "text": "another chat", "chats": [{"guid": "other-chat"}]},
    {"guid": "wrong-parent", "text": "wrong message", "chats": [{"guid": "iMessage;-;fixture"}]},
    {"guid": "parent", "text": "unverified membership"},
    {"guid": "parent", "text": None, "chats": [{"guid": "iMessage;-;fixture"}]},
    httpx.ConnectError("provider unavailable"),
    [],
])
async def test_unavailable_or_unrelated_reply_remains_visible_without_invented_quote(monkeypatch, parent):
    _, handled = await _deliver(monkeypatch, parent)
    event = handled[0]
    assert event.reply_to_text is None
    prepared = GatewayInboundMixin._prepend_inbound_reply_context(event, event.source, event.text)
    assert "could not be loaded" in prepared
    assert prepared.endswith("You didn't do this?")


@pytest.mark.asyncio
async def test_part_guid_is_normalized_and_encoded_without_changing_chat(monkeypatch):
    adapter, handled = await _deliver(monkeypatch, {
        "guid": "parent/+?", "text": "A legitimate partial-message reference.",
        "isFromMe": False, "chats": [{"guid": "iMessage;-;fixture"}],
    }, reference="p:0/parent/+?")
    assert handled[0].reply_to_message_id == "parent/+?"
    assert handled[0].reply_to_text == "A legitimate partial-message reference."
    assert handled[0].reply_to_is_own_message is False
    assert adapter._api_json.await_args.args[1] == "/api/v1/message/parent%2F%2B%3F?with=chats"


@pytest.mark.asyncio
async def test_ordinary_message_does_not_fetch_reply_or_add_context(monkeypatch):
    adapter, handled = await _deliver(monkeypatch, {}, reference=None)
    adapter._api_json.assert_not_awaited()
    event = handled[0]
    assert GatewayInboundMixin._prepend_inbound_reply_context(event, event.source, event.text) == event.text


@pytest.mark.asyncio
async def test_unmentioned_group_does_not_fetch_reply(monkeypatch):
    adapter, handled = await _deliver(monkeypatch, {}, chat="iMessage;+;group", require_mention=True)
    assert handled == []
    adapter._api_json.assert_not_awaited()


@pytest.mark.asyncio
async def test_reply_fetch_cancellation_propagates(monkeypatch):
    with pytest.raises(asyncio.CancelledError):
        await _deliver(monkeypatch, asyncio.CancelledError())
