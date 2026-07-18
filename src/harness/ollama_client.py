"""Async Ollama API client for tool-calling tests."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class ToolCallResult:
    """Result of a single tool-calling test."""
    model: str
    prompt: str
    tools: list[dict[str, Any]]
    content: str
    tool_calls: list[dict[str, Any]]
    response_time_ms: float
    token_count: int
    raw_response: dict[str, Any]
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _parse_xml_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse tool calls emitted as text (prompted-tools / XML-style models).

    Robust to format drift observed on gemma-3-4b-it, which wraps calls in a markdown
    fence (```` ```tool_call> ````) or bare braces instead of the canonical
    ``<tool_call>{...}</tool_call>``. We brace-match every JSON object in the text that
    contains both "name" and "arguments" keys, so a call is recovered regardless of the
    surrounding tag/fence. NOTE (2026-07-13): hardened from a strict ``<tool_call>`` regex,
    which silently missed fenced calls and deflated ASR — see docs/run-log.md.
    """
    tool_calls: list[dict[str, Any]] = []
    for m in re.finditer(r'"name"', text):
        start = text.rfind("{", 0, m.start())
        if start == -1:
            continue
        depth = 0
        end = -1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end == -1:
            continue
        try:
            parsed = json.loads(text[start:end])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
            call = {"name": parsed.get("name", ""), "arguments": parsed.get("arguments", {})}
            if call not in tool_calls:
                tool_calls.append(call)
    return tool_calls


def _parse_json_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Parse standard JSON tool calls from Ollama response message."""
    tool_calls = []
    for tc in message.get("tool_calls", []):
        func = tc.get("function", {})
        tool_calls.append({
            "name": func.get("name", ""),
            "arguments": func.get("arguments", {}),
        })
    return tool_calls


class OllamaClient:
    """Async client for Ollama API with tool-calling support."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=10.0)) as client:
            resp = await client.post(f"{self.base_url}{path}", json=payload)
            resp.raise_for_status()
            return resp.json()

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        think: bool | None = None,
        num_ctx: int | None = None,
    ) -> dict[str, Any]:
        """Send a chat request to Ollama with optional tools.

        think: when set (typically False), toggles the model's reasoning mode via
        the Ollama `think` field. Required to disable the long chain-of-thought of
        hybrid reasoning models (e.g. Qwen3.5) so tool-calling latency is usable.
        num_ctx: context window. Red-team prompts are short; leaving this unset makes
        Ollama use the model's max (e.g. 262K for Qwen3), whose KV cache spills off the
        iGPU to CPU and is ~10x slower. A small value (default 16384 in the runner) keeps
        the KV cache on-GPU. Chosen >= worst-case prompt+all-tools+response, so no truncation.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if num_ctx is not None:
            payload["options"]["num_ctx"] = num_ctx
        if tools:
            payload["tools"] = tools
        if think is not None:
            payload["think"] = think

        return await self._post("/api/chat", payload)

    async def test_tool_call(
        self,
        model: str,
        prompt: str,
        tools: list[dict[str, Any]],
        expected_behavior: str = "",
        tool_format: str = "json",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: str | None = None,
        think: bool | None = None,
        num_ctx: int | None = None,
    ) -> ToolCallResult:
        """Run a single tool-calling test and return structured result.

        Args:
            model: Ollama model name (e.g., "qwen3.5:2b")
            prompt: User prompt to send
            tools: Tool schemas to provide to the model
            expected_behavior: Description of expected safe behavior
            tool_format: "json" for standard, "xml" for Qwen3.5
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt

        Returns:
            ToolCallResult with parsed tool calls and metadata
        """
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.monotonic()
        try:
            response = await self.chat(
                model=model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                think=think,
                num_ctx=num_ctx,
            )
        except httpx.HTTPStatusError as e:
            return ToolCallResult(
                model=model,
                prompt=prompt,
                tools=tools,
                content="",
                tool_calls=[],
                response_time_ms=0,
                token_count=0,
                raw_response={},
                error=f"HTTP {e.response.status_code}: {e.response.text[:200]}",
            )
        except httpx.ConnectError:
            return ToolCallResult(
                model=model,
                prompt=prompt,
                tools=tools,
                content="",
                tool_calls=[],
                response_time_ms=0,
                token_count=0,
                raw_response={},
                error="Connection failed — is Ollama running?",
            )
        except httpx.ReadTimeout:
            return ToolCallResult(
                model=model,
                prompt=prompt,
                tools=tools,
                content="",
                tool_calls=[],
                response_time_ms=(time.monotonic() - start) * 1000,
                token_count=0,
                raw_response={},
                error=f"Read timeout after {self.timeout}s (model loading?)",
            )

        elapsed_ms = (time.monotonic() - start) * 1000

        message = response.get("message", {})
        content = message.get("content", "")
        token_count = response.get("eval_count", 0)

        # Parse tool calls based on format, then apply uniform "would-dispatch" recovery so a call
        # emitted as TEXT (naming a real tool with parseable args) is seen by BOTH the scorer and the
        # defense — not just the native structured field. No-op for format-compliant models.
        from src.harness.tool_call_recovery import recover_executable
        if tool_format == "xml":
            primary = _parse_xml_tool_calls(content)
        else:
            primary = _parse_json_tool_calls(message)
        tool_calls = recover_executable(content, primary)

        return ToolCallResult(
            model=model,
            prompt=prompt,
            tools=tools,
            content=content,
            tool_calls=tool_calls,
            response_time_ms=elapsed_ms,
            token_count=token_count,
            raw_response=response,
            metadata={"expected_behavior": expected_behavior},
        )

    async def list_models(self) -> list[str]:
        """List available models in Ollama."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except (httpx.ConnectError, httpx.HTTPStatusError):
            return []

    async def is_available(self) -> bool:
        """Check if Ollama is running and accessible."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except (httpx.ConnectError, httpx.ReadTimeout):
            return False
