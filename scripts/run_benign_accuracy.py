#!/usr/bin/env python3
"""Benign accuracy measurement for ToolGuard.

Runs all 41 benign prompts through each target model via Ollama and measures:
  - Per-model benign accuracy (% of benign prompts where model calls the right tool)
  - Per-model false positive rate when defense policy is applied to correct calls
  - Domain-level breakdown

Saves results to experiments/benign-accuracy/
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table
from rich import box

from src.attacks.benign_baseline import BenignBaseline, _EXPECTED_TOOLS
from src.defenses.constrained_decoder import ConstrainedDecoder
from src.harness.ollama_client import OllamaClient
from src.harness.tool_schemas import ALL_TOOL_SCHEMAS


MODELS = [
    {"name": "smollm2:1.7b", "tool_format": "json"},
    {"name": "llama3.2:1b", "tool_format": "json"},
    {"name": "llama3.2:3b", "tool_format": "json"},
    {"name": "qwen2.5:3b", "tool_format": "json"},
]

OUTPUT_DIR = Path("experiments/benign-accuracy")
POLICY_PATH = Path("configs/defense_policy.yaml")
OLLAMA_URL = "http://localhost:11434"
TIMEOUT_PER_PROMPT = 60.0  # seconds; skip if exceeded


async def run_benign_for_model(
    client: OllamaClient,
    model_cfg: dict,
    prompts,
    decoder: ConstrainedDecoder,
    console: Console,
) -> dict:
    """Run all benign prompts through one model. Return per-prompt results."""
    model_name = model_cfg["name"]
    tool_format = model_cfg.get("tool_format", "json")

    console.print(f"  [cyan]Loading model: {model_name}...[/cyan]", end=" ")
    try:
        await asyncio.wait_for(
            client.chat(model=model_name, messages=[{"role": "user", "content": "hi"}], max_tokens=1),
            timeout=60.0,
        )
    except Exception as e:
        console.print(f"[red]FAILED to warm up: {e}[/red]")
        return {}
    console.print("[green]ready[/green]")

    results = []
    correct = 0
    fp_by_defense = 0  # correct tool calls that defense would block
    errors = 0

    for p in prompts:
        expected_tools = _EXPECTED_TOOLS.get(p.name, [])

        # Flatten tools for this prompt
        all_tools = []
        if p.tools:
            all_tools = p.tools
        else:
            for tools in ALL_TOOL_SCHEMAS.values():
                all_tools.extend(tools)

        try:
            tc_result = await asyncio.wait_for(
                client.test_tool_call(
                    model=model_name,
                    prompt=p.prompt,
                    tools=all_tools,
                    tool_format=tool_format,
                    temperature=0.0,  # greedy for accuracy measurement
                    max_tokens=512,
                ),
                timeout=TIMEOUT_PER_PROMPT,
            )
            error = tc_result.error
        except asyncio.TimeoutError:
            error = f"Timeout after {TIMEOUT_PER_PROMPT}s"
            tc_result = None

        if error:
            errors += 1
            entry = {
                "prompt_name": p.name,
                "domain": p.metadata.get("domain", ""),
                "edge_case": p.metadata.get("edge_case", False),
                "expected_tools": expected_tools,
                "called_tools": [],
                "correct": False,
                "error": error,
                "defense_blocked": None,
                "violations": [],
            }
            results.append(entry)
            continue

        called_tools = [tc.get("name", "") for tc in (tc_result.tool_calls or [])]
        correct_call = any(name in expected_tools for name in called_tools)

        if correct_call:
            correct += 1

        # Check if defense would block any of the correct tool calls
        defense_blocked = False
        violations = []
        if correct_call and tc_result.tool_calls:
            for tc in tc_result.tool_calls:
                tool_name = tc.get("name", "")
                if tool_name in expected_tools:
                    v = decoder.validate_tool_call(tool_name, tc.get("arguments", {}), user_prompt=p.prompt)
                    if v:
                        defense_blocked = True
                        violations.extend(vi.explanation for vi in v)

        if correct_call and defense_blocked:
            fp_by_defense += 1

        entry = {
            "prompt_name": p.name,
            "prompt": p.prompt,
            "domain": p.metadata.get("domain", ""),
            "edge_case": p.metadata.get("edge_case", False),
            "expected_tools": expected_tools,
            "called_tools": called_tools,
            "model_response": tc_result.content or "",
            "correct": correct_call,
            "error": None,
            "defense_blocked": defense_blocked,
            "violations": violations,
            "response_time_ms": tc_result.response_time_ms,
        }
        results.append(entry)

    total = len(results)
    valid = total - errors
    accuracy = correct / valid if valid > 0 else 0.0
    # FP rate = among correct calls, how many does defense block
    fpr_defense = fp_by_defense / correct if correct > 0 else 0.0

    return {
        "model": model_name,
        "total_prompts": total,
        "errors": errors,
        "correct": correct,
        "accuracy": accuracy,
        "defense_fp_count": fp_by_defense,
        "defense_fpr_of_correct": fpr_defense,
        "results": results,
    }


async def main() -> None:
    console = Console()
    console.print("[bold]ToolGuard Benign Accuracy Measurement[/bold]")
    console.print(f"  Models: {[m['name'] for m in MODELS]}")
    console.print(f"  Output: {OUTPUT_DIR}")
    console.print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = OllamaClient(base_url=OLLAMA_URL, timeout=TIMEOUT_PER_PROMPT + 10.0)
    if not await client.is_available():
        console.print("[red]Ollama is not running at " + OLLAMA_URL + "[/red]")
        sys.exit(1)

    # Load defense policy
    decoder = ConstrainedDecoder.load_policy(POLICY_PATH)
    console.print(f"[green]Policy loaded: {len(decoder.constraints)} constraints.[/green]\n")

    # Generate benign prompts
    baseline = BenignBaseline()
    prompts = baseline.generate_prompts(ALL_TOOL_SCHEMAS)
    console.print(f"Generated {len(prompts)} benign prompts.\n")

    all_model_results = {}

    for model_cfg in MODELS:
        model_name = model_cfg["name"]
        console.print(f"[bold]Running {model_name}...[/bold]")
        model_data = await run_benign_for_model(client, model_cfg, prompts, decoder, console)
        if model_data:
            all_model_results[model_name] = model_data
            acc = model_data["accuracy"]
            fpr = model_data["defense_fpr_of_correct"]
            console.print(
                f"  Accuracy: [{'green' if acc >= 0.7 else 'yellow' if acc >= 0.4 else 'red'}]{acc:.1%}[/]  "
                f"Defense FP rate: [{'green' if fpr == 0 else 'yellow'}]{fpr:.1%}[/]  "
                f"Errors: {model_data['errors']}"
            )
        console.print()

    # Summary table
    table = Table(title="Benign Accuracy Summary", box=box.ROUNDED)
    table.add_column("Model", style="cyan")
    table.add_column("Prompts", justify="right")
    table.add_column("Correct", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Errors", justify="right")
    table.add_column("Defense FPs (of correct)", justify="right", style="yellow")
    table.add_column("Defense FP Rate", justify="right", style="yellow")

    for model_name, data in all_model_results.items():
        acc = data["accuracy"]
        fpr = data["defense_fpr_of_correct"]
        acc_color = "green" if acc >= 0.7 else ("yellow" if acc >= 0.4 else "red")
        table.add_row(
            model_name,
            str(data["total_prompts"]),
            str(data["correct"]),
            f"[{acc_color}]{acc:.1%}[/{acc_color}]",
            str(data["errors"]),
            str(data["defense_fp_count"]),
            f"{fpr:.1%}",
        )

    console.print(table)
    console.print()

    # Domain-level breakdown per model
    domain_table = Table(title="Accuracy by Domain", box=box.ROUNDED)
    domain_table.add_column("Model", style="cyan")
    domain_table.add_column("Domain", style="magenta")
    domain_table.add_column("Correct/Total", justify="right")
    domain_table.add_column("Accuracy", justify="right")

    for model_name, data in all_model_results.items():
        domain_counts: dict[str, dict] = {}
        for r in data["results"]:
            dom = r.get("domain", "unknown")
            if dom not in domain_counts:
                domain_counts[dom] = {"correct": 0, "total": 0}
            domain_counts[dom]["total"] += 1
            if r.get("correct"):
                domain_counts[dom]["correct"] += 1
        for dom in sorted(domain_counts):
            dc = domain_counts[dom]
            acc_val = dc["correct"] / dc["total"] if dc["total"] else 0.0
            acc_color = "green" if acc_val >= 0.7 else ("yellow" if acc_val >= 0.4 else "red")
            domain_table.add_row(
                model_name,
                dom,
                f"{dc['correct']}/{dc['total']}",
                f"[{acc_color}]{acc_val:.1%}[/{acc_color}]",
            )

    console.print(domain_table)

    # Save results
    output_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": [m["name"] for m in MODELS],
        "total_benign_prompts": len(prompts),
        "policy": str(POLICY_PATH),
        "note": (
            "accuracy = fraction of benign prompts where model called the correct tool. "
            "defense_fpr_of_correct = fraction of those correct calls that the defense would block."
        ),
        "model_results": {
            model_name: {k: v for k, v in data.items() if k != "results"}
            for model_name, data in all_model_results.items()
        },
        "detailed_results": {
            model_name: data["results"]
            for model_name, data in all_model_results.items()
        },
    }

    summary_path = OUTPUT_DIR / "results.json"
    with open(summary_path, "w") as f:
        json.dump(output_data, f, indent=2)

    console.print(f"\n[green]Results saved to: {summary_path}[/green]")


if __name__ == "__main__":
    asyncio.run(main())
