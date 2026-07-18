"""HuggingFace Transformers client for tool-calling tests.

Uses tokenizer.apply_chat_template() with native tool support so models
like Qwen3.5 get their trained XML tool-calling format instead of the
JSON/Hermes format that Ollama forces.
"""

from __future__ import annotations

import gc
import json
import re
import sys
import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.harness.ollama_client import ToolCallResult


def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse tool calls from model output.

    Handles Qwen3.5-style XML:
        <tool_call>
        {"name": "func", "arguments": {"key": "value"}}
        </tool_call>
    """
    tool_calls: list[dict[str, Any]] = []
    pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
    for match in re.finditer(pattern, text, re.DOTALL):
        try:
            parsed = json.loads(match.group(1))
            tool_calls.append({
                "name": parsed.get("name", ""),
                "arguments": parsed.get("arguments", {}),
            })
        except json.JSONDecodeError:
            continue
    return tool_calls


class HFClient:
    """Synchronous HuggingFace Transformers client for tool-calling tests.

    Loads models via AutoModelForCausalLM and uses apply_chat_template()
    with native tool schemas so models receive their trained format.
    """

    def __init__(self) -> None:
        self.model: torch.nn.Module | None = None
        self.tokenizer: Any | None = None
        self.current_model_name: str | None = None

    def load_model(self, model_name: str) -> None:
        """Load a model and tokenizer. Unloads previous model if any."""
        if self.current_model_name == model_name and self.model is not None:
            return

        self.unload_model()

        print(f"  Loading {model_name} ...", file=sys.stderr, end=" ", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Load to CPU first, then move to GPU to avoid iGPU hang
        # during streamed weight transfer (device_map="auto" crashes on gfx1103)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        self.model = self.model.to("cuda")
        torch.cuda.synchronize()
        self.model.eval()
        self.current_model_name = model_name
        print("ready", file=sys.stderr, flush=True)

    def unload_model(self) -> None:
        """Unload model and free GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.current_model_name = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_new_tokens: int = 2048,
    ) -> str:
        """Generate a response using apply_chat_template with tools."""
        assert self.model is not None and self.tokenizer is not None, "Model not loaded"

        text = self.tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
            )

        # Decode only the generated tokens (skip the prompt)
        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Free intermediate tensors to prevent VRAM fragmentation on iGPU
        del output_ids, inputs
        torch.cuda.empty_cache()

        return result

    def test_tool_call(
        self,
        model: str,
        prompt: str,
        tools: list[dict[str, Any]],
        expected_behavior: str = "",
        tool_format: str = "native",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        system_prompt: str | None = None,
    ) -> ToolCallResult:
        """Run a single tool-calling test and return structured result.

        Same interface as OllamaClient.test_tool_call() for drop-in use.
        """
        self.load_model(model)

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        start = time.monotonic()
        try:
            content = self.generate(
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_new_tokens=max_tokens,
            )
        except Exception as e:
            return ToolCallResult(
                model=model,
                prompt=prompt,
                tools=tools,
                content="",
                tool_calls=[],
                response_time_ms=(time.monotonic() - start) * 1000,
                token_count=0,
                raw_response={},
                error=str(e),
            )

        elapsed_ms = (time.monotonic() - start) * 1000
        tool_calls = _parse_tool_calls(content)
        token_count = len(self.tokenizer.encode(content))

        return ToolCallResult(
            model=model,
            prompt=prompt,
            tools=tools,
            content=content,
            tool_calls=tool_calls,
            response_time_ms=elapsed_ms,
            token_count=token_count,
            raw_response={"generated_text": content},
            metadata={"expected_behavior": expected_behavior},
        )
