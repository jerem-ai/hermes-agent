"""Resolve native reply pointers through the owning BlueBubbles adapter."""
import asyncio
import logging
import re
from urllib.parse import quote

import httpx

logger = logging.getLogger(__name__)


async def resolve_reply(adapter, record, chat_guid):
    """Return (message id, quoted text, own message), requiring exact chat membership."""
    reference = adapter._value(record.get("threadOriginatorGuid"), record.get("associatedMessageGuid"))
    if not reference:
        return None, None, False
    # iMessage associated-message references can address a message part.
    reference = re.sub(r"^(?:p|bp):\d+/", "", reference)
    if not chat_guid:
        return reference, None, False
    try:
        response = await asyncio.wait_for(adapter._api_json(
            "get", f"/api/v1/message/{quote(reference, safe='')}?with=chats", timeout=5,
        ), timeout=5)
        parent = response.get("data") if isinstance(response, dict) else None
        if not isinstance(parent, dict) or parent.get("guid") != reference:
            return reference, None, False
        chats = parent.get("chats")
        if not isinstance(chats, list) or not any(
            isinstance(chat, dict) and chat.get("guid") == chat_guid for chat in chats
        ):
            return reference, None, False
        text = parent.get("text")
        return reference, text if isinstance(text, str) and text else None, parent.get("isFromMe") is True
    except (httpx.HTTPError, TimeoutError, ValueError) as exc:
        # Exception URLs can contain the server password. Keep credentials and message text out of logs.
        logger.warning("[bluebubbles] reply lookup unavailable (%s)", type(exc).__name__)
        return reference, None, False
