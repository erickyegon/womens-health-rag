"""Small helpers for working with LLM chat messages."""

from __future__ import annotations

from langchain_core.messages import BaseMessage


def message_text(message: BaseMessage) -> str:
    """Return the text of a chat message.

    ``content`` is a plain string for normal chat completions, but the type allows a
    list of content blocks. In that case the text blocks are joined.
    """
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)
