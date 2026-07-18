"""Uniform "would-dispatch" tool-call recovery (TMLR revision).

The original harness read tool calls ONLY from the model's native structured field. Weaker models
(e.g. Llama-3.2-1B) frequently emit a well-formed tool call as TEXT instead — naming a real tool with
parseable arguments, often using ``parameters``/``function`` key aliases or a markdown fence. Such a call
WOULD dispatch through a permissive tool runner, so both the scorer and the ToolGuard defense must see it,
or they systematically under-count attacks (and the defense waves the attack through).

This module defines the executability tiers used uniformly for every model:
  - strict-native      : the native structured field only (what the original harness saw).
  - recovered-executable: native ∪ text-emitted calls naming a REAL schema tool with parseable args. HEADLINE.
  - intent-upper-bound : native ∪ ANY text-emitted call, including hallucinated/non-schema tool names.
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.harness.tool_schemas import ALL_TOOL_SCHEMAS

REAL_SCHEMA_TOOLS: frozenset[str] = frozenset(
    t["function"]["name"] for ts in ALL_TOOL_SCHEMAS.values() for t in ts
)


def extract_text_calls(text: str) -> list[dict[str, Any]]:
    """Recover every tool-call-shaped JSON object in free text.

    Accepts ``name``/``function`` for the tool name and ``arguments``/``parameters`` for the args,
    regardless of surrounding tag or markdown fence. Returns normalized {name, arguments} dicts.
    """
    if not text:
        return []
    calls: list[dict[str, Any]] = []
    for key in ('"name"', '"function"'):
        for m in re.finditer(re.escape(key), text):
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
                obj = json.loads(text[start:end])
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            name = obj.get("name") or obj.get("function")
            args = obj.get("arguments")
            if args is None:
                args = obj.get("parameters", {})
            if name and isinstance(name, str):
                call = {"name": name, "arguments": args if isinstance(args, dict) else {}}
                if call not in calls:
                    calls.append(call)
    return calls


def _merge(base: list[dict[str, Any]], extra: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = list(base)
    for c in extra:
        if c not in out:
            out.append(c)
    return out


def recover_executable(text: str, native_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """HEADLINE tier: native ∪ text calls naming a REAL schema tool (would dispatch)."""
    extra = [c for c in extract_text_calls(text) if c.get("name") in REAL_SCHEMA_TOOLS]
    return _merge(native_calls, extra)


def recover_intent(text: str, native_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """UPPER BOUND tier: native ∪ ANY text call, including hallucinated tool names."""
    return _merge(native_calls, extract_text_calls(text))
