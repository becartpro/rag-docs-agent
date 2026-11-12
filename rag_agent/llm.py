"""LLM clients behind one small interface, so the agent does not depend on a vendor.

Each client receives the conversation and the tool schemas and returns either a final answer
or a list of tool calls. Anthropic is implemented with its native tool-use API; a FakeLLM is
provided for tests and offline demos.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMReply:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLM(Protocol):
    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMReply: ...


class AnthropicLLM:
    """Minimal client for the Anthropic Messages API with tool use."""

    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: str | None = None, api_key: str | None = None, timeout: float = 60):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
        self.client = httpx.Client(timeout=timeout)

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMReply:
        payload = {"model": self.model, "max_tokens": 1024, "system": system,
                   "messages": messages, "tools": tools}
        headers = {"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
        resp = self.client.post(self.URL, headers=headers, content=json.dumps(payload))
        resp.raise_for_status()
        reply = LLMReply()
        for block in resp.json().get("content", []):
            if block["type"] == "text":
                reply.text += block["text"]
            elif block["type"] == "tool_use":
                reply.tool_calls.append(ToolCall(block["id"], block["name"], block["input"]))
        return reply


class FakeLLM:
    """Scripted LLM: returns the replies it was given, one per call (for tests and demos)."""

    def __init__(self, replies: list[LLMReply]):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMReply:
        self.calls.append([dict(m) for m in messages])
        if not self.replies:
            raise RuntimeError("FakeLLM ran out of scripted replies")
        return self.replies.pop(0)
