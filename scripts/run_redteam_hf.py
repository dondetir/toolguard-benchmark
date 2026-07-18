#!/usr/bin/env python3
"""Entry point for ToolGuard red-teaming using HuggingFace Transformers.

Designed to run INSIDE the ROCm container. Uses native tool-calling
format via apply_chat_template() instead of Ollama's JSON/Hermes format.

Usage:
    ~/rocm-base/run.sh python scripts/run_redteam_hf.py \
        --model Qwen/Qwen3.5-0.8B --attack all --runs 3
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.attacks import Attack, AttackResult, AttackCategory
from src.attacks.chain_attacks import ChainAttack
from src.attacks.data_exfiltration import DataExfiltrationAttack
from src.attacks.parameter_injection import ParameterInjectionAttack
from src.attacks.privilege_escalation import PrivilegeEscalationAttack
from src.attacks.tool_substitution import ToolSubstitutionAttack
from src.harness.hf_client import HFClient
from src.harness.tool_schemas import get_tools_for_domains

import yaml

ATTACK_REGISTRY: dict[str, type[Attack]] = {
    "parameter_injection": ParameterInjectionAttack,
    "tool_substitution": ToolSubstitutionAttack,
    "privilege_escalation": PrivilegeEscalationAttack,
    "data_exfiltration": DataExfiltrationAttack,
    "chain_attacks": ChainAttack,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ToolGuard HF Red-Teaming Runner")
    parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model ID (e.g., 'Qwen/Qwen3.5-0.8B')",
    )
    parser.add_argument(
        "--attack", type=str, default="all",
        help="Attack category or 'all' (default: all)",
    )
    parser.add_argument(
        "--runs", type=int, default=3,
        help="Number of runs per attack prompt (default: 3)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: experiments/YYYYMMDD-HHMMSS/)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2048,
        help="Max new tokens to generate (default: 2048)",
    )
    parser.add_argument(
        "--attack-config", type=str, default="configs/attacks.yaml",
        help="Attack config path (default: configs/attacks.yaml)",
    )
    return parser.parse_args()


def _serialize_result(result: AttackResult) -> dict[str, Any]:
    d = asdict(result)
    d["category"] = result.category.value
    d["severity"] = result.severity.value
    return d


def main() -> None:
    args = parse_args()

    print(f"ToolGuard HF Red-Teaming Runner")
    print(f"  Model:       {args.model}")
    print(f"  Attack:      {args.attack}")
    print(f"  Runs:        {args.runs}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Max tokens:  {args.max_tokens}")
    print()

    # Load attack config
    with open(args.attack_config) as f:
        attack_config = yaml.safe_load(f)

    categories = attack_config.get("attack_categories", {})
    if args.attack == "all":
        attack_names = [name for name, cfg in categories.items() if cfg.get("enabled", False)]
    else:
        if args.attack in categories and categories[args.attack].get("enabled", False):
            attack_names = [args.attack]
        else:
            print(f"Error: unknown or disabled attack: {args.attack}", file=sys.stderr)
            sys.exit(1)

    if not attack_names:
        print("Error: no enabled attacks found", file=sys.stderr)
        sys.exit(1)

    # Set up output
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output or f"experiments/hf-{ts}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create client and load model
    client = HFClient()
    try:
        client.load_model(args.model)
    except Exception as exc:
        error_msg = f"Fatal: failed to load model {args.model}: {exc}"
        print(f"\n{error_msg}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        # Write a minimal results.json so the output dir is never silently empty
        error_summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "provider": "hf",
            "fatal_error": error_msg,
            "attacks": attack_names,
            "results": {},
        }
        with open(output_dir / "results.json", "w") as f:
            json.dump(error_summary, f, indent=2)
        print(f"Error details written to: {output_dir / 'results.json'}", file=sys.stderr)
        sys.exit(1)

    all_results: dict[str, list[dict[str, Any]]] = {}

    for attack_name in attack_names:
        print(f"\n=== Attack: {attack_name} ===")
        attack_cfg = categories[attack_name]
        attack_cls = ATTACK_REGISTRY.get(attack_name)
        if not attack_cls:
            print(f"  Skipping {attack_name}: no implementation", file=sys.stderr)
            continue

        attack = attack_cls()
        domains = attack_cfg.get("domains", ["all"])
        tool_schemas = get_tools_for_domains(domains)
        prompts = attack.generate_prompts(tool_schemas)

        all_tools: list[dict[str, Any]] = []
        for tools in tool_schemas.values():
            all_tools.extend(tools)

        results: list[dict[str, Any]] = []
        for i, prompt in enumerate(prompts):
            for run_idx in range(args.runs):
                print(f"  [{i+1}/{len(prompts)}] run {run_idx+1}/{args.runs}: {prompt.name[:50]}...", end=" ", flush=True)

                tc_result = client.test_tool_call(
                    model=args.model,
                    prompt=prompt.prompt,
                    tools=prompt.tools or all_tools,
                    expected_behavior=prompt.expected_safe_behavior,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
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
                    result = attack.evaluate_response(response_dict, prompt)
                    result.metadata["run"] = run_idx
                    result.metadata["response_time_ms"] = tc_result.response_time_ms
                    result.metadata["token_count"] = tc_result.token_count

                status = "VULN" if result.success else "SAFE"
                print(f"{status} ({tc_result.response_time_ms:.0f}ms)")
                results.append(_serialize_result(result))

        all_results[attack_name] = results

    # Unload model
    client.unload_model()

    # Save results
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "provider": "hf",
        "attacks": attack_names,
        "runs_per_attack": args.runs,
        "temperature": args.temperature,
        "results": {args.model: all_results},
    }
    output_file = output_dir / "results.json"
    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Results Summary: {args.model}")
    print(f"{'='*60}")
    for attack_name, results in all_results.items():
        total = len(results)
        succeeded = sum(1 for r in results if r.get("success"))
        asr = succeeded / total if total > 0 else 0.0
        print(f"  {attack_name:30s}  {succeeded}/{total}  ASR={asr:.1%}")

    total_all = sum(len(r) for r in all_results.values())
    succ_all = sum(sum(1 for r in res if r.get("success")) for res in all_results.values())
    overall_asr = succ_all / total_all if total_all > 0 else 0.0
    print(f"  {'OVERALL':30s}  {succ_all}/{total_all}  ASR={overall_asr:.1%}")
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
