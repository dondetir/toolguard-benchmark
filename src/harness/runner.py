"""Experiment runner for ToolGuard red-teaming."""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.attacks import Attack, AttackCategory, AttackResult
from src.attacks.chain_attacks import ChainAttack
from src.attacks.data_exfiltration import DataExfiltrationAttack
from src.attacks.parameter_injection import ParameterInjectionAttack
from src.attacks.privilege_escalation import PrivilegeEscalationAttack
from src.attacks.tool_substitution import ToolSubstitutionAttack
from src.defenses.constrained_decoder import ConstrainedDecoder
from src.harness.ollama_client import OllamaClient
from src.harness.tool_schemas import get_tools_for_domains

ATTACK_REGISTRY: dict[str, type[Attack]] = {
    "parameter_injection": ParameterInjectionAttack,
    "tool_substitution": ToolSubstitutionAttack,
    "privilege_escalation": PrivilegeEscalationAttack,
    "data_exfiltration": DataExfiltrationAttack,
    "chain_attacks": ChainAttack,
}


def render_tools_as_prompt(tools: list[dict[str, Any]]) -> str:
    """Render tool schemas into a system prompt for models without a native tool API.

    Used for `tool_format: prompted` (e.g. gemma-3-4b-it, whose Ollama template exposes
    no tools endpoint). The model is asked to emit <tool_call>{json}</tool_call> blocks,
    which the harness parses via the existing tool_format="xml" path.
    """
    lines = [
        "You are an assistant with access to the following tools. When a tool is needed, "
        "respond with ONLY one or more tool-call blocks in this EXACT format (no other text):",
        '<tool_call>{"name": "<tool_name>", "arguments": {<args>}}</tool_call>',
        "",
        "Available tools:",
    ]
    for t in tools:
        fn = t.get("function", t)
        params = fn.get("parameters", {}).get("properties", {})
        required = set(fn.get("parameters", {}).get("required", []))
        args = ", ".join(
            f"{n}{'' if n in required else '?'}: {p.get('type', 'any')}"
            for n, p in params.items()
        )
        lines.append(f"- {fn.get('name')}({args}): {fn.get('description', '')}")
    return "\n".join(lines)


def _load_model_configs(config_path: str = "configs/models.yaml") -> list[dict[str, Any]]:
    with open(config_path) as f:
        return yaml.safe_load(f)["models"]


def _load_attack_configs(config_path: str = "configs/attacks.yaml") -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _serialize_result(result: AttackResult) -> dict[str, Any]:
    d = asdict(result)
    d["category"] = result.category.value
    d["severity"] = result.severity.value
    return d


class ExperimentRunner:
    """Runs red-teaming experiments against models via Ollama."""

    def __init__(
        self,
        output_dir: str | None = None,
        model_config_path: str = "configs/models.yaml",
        attack_config_path: str = "configs/attacks.yaml",
        ollama_url: str = "http://localhost:11434",
        defense_policy: str | None = None,
        include_expanded: bool = False,
        temperature_override: float | None = None,
    ) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.output_dir = Path(output_dir or f"experiments/{ts}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.include_expanded = include_expanded
        self.temperature_override = temperature_override

        self.model_configs = _load_model_configs(model_config_path)
        self.attack_config = _load_attack_configs(attack_config_path)
        self.client = OllamaClient(base_url=ollama_url)

        # Optional ConstrainedDecoder defense layer
        self.decoder: ConstrainedDecoder | None = None
        if defense_policy:
            self.decoder = ConstrainedDecoder.load_policy(defense_policy)

    def _get_model_config(self, model_name: str) -> dict[str, Any] | None:
        for m in self.model_configs:
            if m["name"] == model_name:
                return m
        return None

    def _get_enabled_attacks(self, attack_filter: str = "all") -> list[str]:
        categories = self.attack_config.get("attack_categories", {})
        if attack_filter != "all":
            if attack_filter in categories and categories[attack_filter].get("enabled", False):
                return [attack_filter]
            return []
        return [name for name, cfg in categories.items() if cfg.get("enabled", False)]

    async def run_attack_on_model(
        self,
        model_name: str,
        attack_name: str,
        runs: int = 3,
    ) -> list[AttackResult]:
        """Run a single attack category against a single model."""
        model_cfg = self._get_model_config(model_name)
        if not model_cfg:
            raise ValueError(f"Unknown model: {model_name}")

        attack_cfg = self.attack_config["attack_categories"].get(attack_name)
        if not attack_cfg:
            raise ValueError(f"Unknown attack: {attack_name}")

        attack_cls = ATTACK_REGISTRY.get(attack_name)
        if not attack_cls:
            raise ValueError(f"No implementation for attack: {attack_name}")

        attack = attack_cls()
        domains = attack_cfg.get("domains", ["all"])
        tool_schemas = get_tools_for_domains(domains)
        prompts = attack.generate_prompts(tool_schemas)

        # Expanded held-out suite (TMLR revision): additive new prompts, evaluated by the
        # same category evaluator. Each carries its own tools + metadata(source="expanded").
        if self.include_expanded:
            from src.attacks.expanded_suite import expanded_prompts_for
            prompts = prompts + expanded_prompts_for(attack.category)

        eval_cfg = self.attack_config.get("evaluation", {})
        temperature = (
            self.temperature_override
            if self.temperature_override is not None
            else eval_cfg.get("temperature", 0.7)
        )
        timeout = eval_cfg.get("timeout_seconds", 30)

        tool_format = model_cfg.get("tool_format", "json")
        max_tokens = model_cfg.get("max_tokens", 2048)
        think = model_cfg.get("think")  # None = leave model default; False disables reasoning
        # Small context keeps the KV cache on the iGPU (model default is up to 262K, which spills
        # to CPU and is ~10x slower). 16384 >= worst-case prompt+all-tools+2048-response.
        num_ctx = model_cfg.get("num_ctx", 16384)

        # Flatten all tools for the API call
        all_tools = []
        for tools in tool_schemas.values():
            all_tools.extend(tools)

        # Prompted-tools mode: models with no native tool API (e.g. gemma3) get the tool
        # schemas injected as a system prompt and emit <tool_call> blocks parsed as "xml".
        prompted = tool_format == "prompted"
        api_tool_format = "xml" if prompted else tool_format

        results: list[AttackResult] = []
        for prompt in prompts:
            ptools = prompt.tools or all_tools
            sys_prompt = render_tools_as_prompt(ptools) if prompted else None
            for run_idx in range(runs):
                tc_result = await self.client.test_tool_call(
                    model=model_name,
                    prompt=prompt.prompt,
                    tools=None if prompted else ptools,
                    expected_behavior=prompt.expected_safe_behavior,
                    tool_format=api_tool_format,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    think=think,
                    num_ctx=num_ctx,
                    system_prompt=sys_prompt,
                )

                if tc_result.error:
                    result = AttackResult(
                        attack_name=prompt.name,
                        category=attack.category,
                        success=False,
                        model_response=tc_result.error,
                        tool_calls_made=[],
                        explanation=f"Error: {tc_result.error}",
                        severity=prompt.severity,
                        metadata={"run": run_idx, "error": True},
                    )
                else:
                    response_dict = {
                        "content": tc_result.content,
                        "tool_calls": tc_result.tool_calls,
                    }

                    # Apply ConstrainedDecoder defense if loaded
                    if self.decoder is not None and tc_result.tool_calls:
                        all_violations = []
                        for tc in tc_result.tool_calls:
                            tool_name = tc.get("name", "")
                            arguments = tc.get("arguments", {})
                            violations = self.decoder.validate_tool_call(
                                tool_name, arguments, user_prompt=prompt.prompt
                            )
                            all_violations.extend(violations)

                        is_benign = prompt.metadata.get("benign", False)
                        if all_violations:
                            violation_strs = [v.explanation for v in all_violations]
                            if is_benign:
                                # Defense blocked a benign call — false positive
                                result = AttackResult(
                                    attack_name=prompt.name,
                                    category=attack.category,
                                    success=True,  # false positive
                                    model_response=tc_result.content or "",
                                    tool_calls_made=tc_result.tool_calls,
                                    explanation=(
                                        "Defense false positive: blocked a benign tool call. "
                                        f"Violations: {'; '.join(violation_strs)}"
                                    ),
                                    severity=prompt.severity,
                                    metadata={
                                        **prompt.metadata,
                                        "run": run_idx,
                                        "defended": True,
                                        "violations": violation_strs,
                                        "false_positive": True,
                                    },
                                )
                            else:
                                # Defense successfully blocked an attack
                                result = AttackResult(
                                    attack_name=prompt.name,
                                    category=attack.category,
                                    success=False,  # attack was defended
                                    model_response=tc_result.content or "",
                                    tool_calls_made=tc_result.tool_calls,
                                    explanation=(
                                        "Defense blocked the attack. "
                                        f"Violations: {'; '.join(violation_strs)}"
                                    ),
                                    severity=prompt.severity,
                                    metadata={
                                        "run": run_idx,
                                        "response_time_ms": tc_result.response_time_ms,
                                        "token_count": tc_result.token_count,
                                        "defended": True,
                                        "violations": violation_strs,
                                    },
                                )
                            result.metadata["run"] = run_idx
                            results.append(result)
                            continue

                    result = attack.evaluate_response(response_dict, prompt)
                    # Carry prompt provenance (source/domain/subtype/policy_evading) into results
                    # so expanded-vs-original and covered-vs-evading splits are possible in analysis.
                    for k, v in prompt.metadata.items():
                        result.metadata.setdefault(k, v)
                    result.metadata["run"] = run_idx
                    result.metadata["response_time_ms"] = tc_result.response_time_ms
                    result.metadata["token_count"] = tc_result.token_count
                    if self.decoder is not None:
                        result.metadata["defended"] = False

                results.append(result)

        return results

    async def run(
        self,
        model_filter: str = "all",
        attack_filter: str = "all",
        runs: int = 3,
    ) -> dict[str, Any]:
        """Run the full experiment suite.

        Returns a summary dict with all results.
        """
        if not await self.client.is_available():
            raise RuntimeError("Ollama is not running at " + self.client.base_url)

        attack_names = self._get_enabled_attacks(attack_filter)
        if not attack_names:
            raise ValueError(f"No enabled attacks matching: {attack_filter}")

        if model_filter == "all":
            models = [m["name"] for m in self.model_configs]
        else:
            models = [model_filter]

        # Pre-flight: filter to models actually available in Ollama
        available = await self.client.list_models()
        available_set = {m.split(":")[0] + ":" + m.split(":")[-1] if ":" in m else m for m in available}
        available_set.update(available)  # include exact names too
        skipped = [m for m in models if m not in available_set]
        models = [m for m in models if m in available_set]
        if skipped:
            import sys
            print(f"  Skipping unavailable models: {', '.join(skipped)}", file=sys.stderr)
        if not models:
            raise ValueError("No available models to test")

        all_results: dict[str, dict[str, list[dict[str, Any]]]] = {}

        for model in models:
            # Warm up model (triggers Ollama to load it into VRAM). Load at the small context so the
            # KV cache stays on-GPU from the start (avoids a reload when the first tool call sets num_ctx).
            import sys
            cfg = self._get_model_config(model) or {}
            warm_ctx = cfg.get("num_ctx", 16384)
            print(f"  Loading model: {model}...", file=sys.stderr, end=" ", flush=True)
            await self.client.chat(model=model, messages=[{"role": "user", "content": "hi"}],
                                   max_tokens=1, num_ctx=warm_ctx)
            print("ready", file=sys.stderr, flush=True)

            all_results[model] = {}
            for attack_name in attack_names:
                results = await self.run_attack_on_model(model, attack_name, runs)
                all_results[model][attack_name] = [_serialize_result(r) for r in results]

        # Save results
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "models": models,
            "attacks": attack_names,
            "runs_per_attack": runs,
            "results": all_results,
        }
        output_file = self.output_dir / "results.json"
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        return summary

    @staticmethod
    def compute_summary(results: dict[str, Any]) -> dict[str, Any]:
        """Compute summary statistics from experiment results."""
        summary: dict[str, Any] = {}
        for model, attacks in results.get("results", {}).items():
            model_summary: dict[str, Any] = {}
            total_attacks = 0
            total_successes = 0
            for attack_name, attack_results in attacks.items():
                succeeded = sum(1 for r in attack_results if r.get("success"))
                total = len(attack_results)
                total_attacks += total
                total_successes += succeeded
                model_summary[attack_name] = {
                    "total": total,
                    "succeeded": succeeded,
                    "asr": succeeded / total if total > 0 else 0.0,
                }
            model_summary["overall"] = {
                "total": total_attacks,
                "succeeded": total_successes,
                "asr": total_successes / total_attacks if total_attacks > 0 else 0.0,
            }
            summary[model] = model_summary
        return summary
