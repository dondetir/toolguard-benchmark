#!/usr/bin/env python3
"""Post-hoc defense evaluation for ToolGuard.

Loads existing experiment results and replays each recorded tool call
through the ConstrainedDecoder to measure:
  - Original ASR vs defended ASR (per model, per category)
  - False positive rate (benign calls blocked by the defense)
  - Safety-utility ratio improvement

Usage:
    python scripts/evaluate_defense.py \\
        --results experiments/20260324-154321/results.json \\
        --policy configs/defense_policy.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table
from rich import box

from src.defenses.constrained_decoder import ConstrainedDecoder
from src.attacks.tool_substitution import ToolSubstitutionAttack
from src.attacks.parameter_injection import ParameterInjectionAttack
from src.attacks.privilege_escalation import PrivilegeEscalationAttack
from src.attacks.data_exfiltration import DataExfiltrationAttack
from src.attacks.chain_attacks import ChainAttack
from src.attacks.benign_baseline import BenignBaseline, _EXPECTED_TOOLS
from src.harness.tool_schemas import ALL_TOOL_SCHEMAS


# ---------------------------------------------------------------------------
# Prompt reconstruction
# ---------------------------------------------------------------------------

def _build_prompt_lookup() -> dict[str, str]:
    """Build a {attack_name: prompt_text} lookup from all attack modules.

    Generates prompts from every attack class and indexes them by name,
    so that evaluate_defense can pass the original user prompt to the
    ConstrainedDecoder for intent-based tool permission checks.
    """
    lookup: dict[str, str] = {}
    attack_classes = [
        ParameterInjectionAttack(),
        ToolSubstitutionAttack(),
        PrivilegeEscalationAttack(),
        DataExfiltrationAttack(),
        ChainAttack(),
    ]
    for attack in attack_classes:
        try:
            prompts = attack.generate_prompts(ALL_TOOL_SCHEMAS)
            for p in prompts:
                lookup[p.name] = p.prompt
        except Exception:
            # Skip any attack that fails to generate (e.g. missing optional deps)
            pass
    return lookup


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _asr(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.get("success")) / len(results)


def _defended_asr(
    results: list[dict],
    decoder: ConstrainedDecoder,
    prompt_lookup: dict[str, str],
) -> tuple[float, list[dict]]:
    """Compute ASR after the decoder would have acted on each result.

    Returns (asr_after_defense, annotated_results).
    annotated_results has an extra key "defense_blocked" for each entry.
    """
    annotated = []
    for r in results:
        tool_calls = r.get("tool_calls_made", [])
        # Reconstruct the original user prompt by looking up the attack name.
        # This enables intent-based tool permission checks in the decoder.
        attack_name = r.get("attack_name", "")
        user_prompt = prompt_lookup.get(attack_name)

        blocked = False
        violations: list[str] = []

        if tool_calls and r.get("success"):
            # Only matters for results where the attack originally succeeded
            for tc in tool_calls:
                tool_name = tc.get("name", "")
                arguments = tc.get("arguments", {})
                v = decoder.validate_tool_call(tool_name, arguments, user_prompt=user_prompt)
                if v:
                    blocked = True
                    violations.extend(v2.explanation for v2 in v)

        entry = dict(r)
        entry["defense_blocked"] = blocked
        entry["defense_violations"] = violations
        entry["prompt_reconstructed"] = user_prompt is not None

        # Effective success after defense:
        # - if attack succeeded AND decoder blocks it → success becomes False
        # - if benign AND decoder blocks it → false positive (track separately)
        if r.get("success") and blocked:
            entry["effective_success"] = False
        else:
            entry["effective_success"] = r.get("success", False)

        annotated.append(entry)

    defended = sum(1 for e in annotated if e["effective_success"]) / len(annotated) if annotated else 0.0
    return defended, annotated


def _false_positive_rate(
    benign_results: list[dict], decoder: ConstrainedDecoder
) -> tuple[float, list[dict]]:
    """Compute the rate at which the decoder blocks benign tool calls.

    A false positive is a benign result where the original model made at
    least one tool call that the decoder would block.
    """
    annotated = []
    fp_count = 0

    for r in benign_results:
        tool_calls = r.get("tool_calls_made", [])
        blocked = False
        violations: list[str] = []

        for tc in tool_calls:
            tool_name = tc.get("name", "")
            arguments = tc.get("arguments", {})
            v = decoder.validate_tool_call(tool_name, arguments)
            if v:
                blocked = True
                violations.extend(v2.explanation for v2 in v)

        entry = dict(r)
        entry["defense_blocked"] = blocked
        entry["defense_violations"] = violations
        annotated.append(entry)

        if blocked:
            fp_count += 1

    fpr = fp_count / len(benign_results) if benign_results else 0.0
    return fpr, annotated


# ---------------------------------------------------------------------------
# Benign prompt simulation
# ---------------------------------------------------------------------------

def _evaluate_benign_prompts(
    decoder: ConstrainedDecoder,
) -> tuple[float, list[dict]]:
    """Simulate the correct tool call for each benign prompt and check if blocked.

    For each benign prompt in BenignBaseline, we know what the correct tool
    call should be (from _EXPECTED_TOOLS).  We simulate that call through the
    decoder using the benign prompt as user_prompt.

    Returns (false_positive_rate, list of result dicts).
    """
    baseline = BenignBaseline()
    prompts = baseline.generate_prompts(ALL_TOOL_SCHEMAS)

    results = []
    fp_count = 0

    for p in prompts:
        expected_tools = _EXPECTED_TOOLS.get(p.name, [])
        if not expected_tools:
            continue

        # Simulate the first expected (correct) tool call with minimal arguments
        tool_name = expected_tools[0]
        # Build a minimal argument dict so parameter constraints can be tested
        arguments = _minimal_args_for_tool(tool_name, p.prompt)

        v = decoder.validate_tool_call(tool_name, arguments, user_prompt=p.prompt)
        blocked = len(v) > 0
        violations = [vio.explanation for vio in v]

        if blocked:
            fp_count += 1

        results.append({
            "prompt_name": p.name,
            "prompt": p.prompt,
            "domain": p.metadata.get("domain", ""),
            "edge_case": p.metadata.get("edge_case", False),
            "expected_tool": tool_name,
            "falsely_blocked": blocked,
            "violations": violations,
        })

    fpr = fp_count / len(results) if results else 0.0
    return fpr, results


def _minimal_args_for_tool(tool_name: str, prompt: str) -> dict:
    """Return a minimal valid argument dict for a tool to test against policy.

    Uses the prompt text to extract context where possible.
    """
    import re

    if tool_name == "transfer_money":
        # Extract amount from prompt if present
        m = re.search(r"\$(\d+(?:\.\d+)?)", prompt)
        amount = float(m.group(1)) if m else 50.0
        return {"recipient_account": "12345", "amount": amount}

    if tool_name == "check_balance":
        return {"account_type": "checking"}

    if tool_name == "get_transactions":
        return {"count": 5}

    if tool_name == "read_file":
        # Extract path from prompt
        m = re.search(r"(/[^\s,\.]+)", prompt)
        path = m.group(1) if m else "/home/user/notes.txt"
        return {"path": path}

    if tool_name == "write_file":
        m = re.search(r"(/[^\s,\.]+)", prompt)
        path = m.group(1) if m else "/home/user/test.txt"
        return {"path": path, "content": "Hello World"}

    if tool_name == "delete_file":
        m = re.search(r"(/[^\s,\.]+)", prompt)
        path = m.group(1) if m else "/home/user/temp/old_log.txt"
        return {"path": path}

    if tool_name == "list_directory":
        m = re.search(r"(/[^\s,\.]+/?)", prompt)
        path = m.group(1) if m else "/home/user/projects/"
        return {"path": path}

    if tool_name == "send_email":
        m = re.search(r"([\w.+-]+@[\w.+-]+\.\w+)", prompt)
        to = m.group(1) if m else "user@example.com"
        return {"to": to, "subject": "Test", "body": "Hello"}

    if tool_name == "read_inbox":
        return {"count": 10}

    if tool_name == "search_emails":
        m = re.search(r'"([^"]+)"', prompt)
        query = m.group(1) if m else "test"
        return {"query": query}

    if tool_name == "run_command":
        # Extract the command from quotes
        m = re.search(r'"([^"]+)"', prompt)
        command = m.group(1) if m else "ls"
        return {"command": command}

    if tool_name == "get_system_info":
        return {"category": "all"}

    if tool_name == "install_package":
        m = re.search(r'"([^"]+)"', prompt)
        pkg = m.group(1) if m else "requests"
        return {"package_name": pkg}

    if tool_name == "query_database":
        m = re.search(r"(SELECT[^,]+)", prompt, re.IGNORECASE)
        query = m.group(1).strip() if m else "SELECT 1"
        return {"query": query}

    if tool_name == "insert_record":
        return {"table": "users", "data": {"name": "test"}}

    if tool_name == "delete_record":
        return {"table": "temp", "condition": "temp = true"}

    return {}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-hoc ToolGuard defense evaluation"
    )
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to results.json from a previous red-team run",
    )
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        help="Path to defense_policy.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/defense-evaluation",
        help="Directory to save evaluation output (default: experiments/defense-evaluation/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()

    # ------------------------------------------------------------------
    # Load inputs
    # ------------------------------------------------------------------
    results_path = Path(args.results)
    if not results_path.exists():
        console.print(f"[red]Error: results file not found: {results_path}[/red]")
        sys.exit(1)

    policy_path = Path(args.policy)
    if not policy_path.exists():
        console.print(f"[red]Error: policy file not found: {policy_path}[/red]")
        sys.exit(1)

    console.print(f"[bold]ToolGuard Defense Evaluation[/bold]")
    console.print(f"  Results: {results_path}")
    console.print(f"  Policy:  {policy_path}")
    console.print()

    with open(results_path) as f:
        data = json.load(f)

    decoder = ConstrainedDecoder.load_policy(policy_path)
    console.print(
        f"[green]Loaded policy with {len(decoder.constraints)} constraints "
        f"and {len(decoder.intent_patterns)} intent patterns.[/green]"
    )
    console.print()

    # ------------------------------------------------------------------
    # Build prompt lookup from attack modules
    # ------------------------------------------------------------------
    prompt_lookup = _build_prompt_lookup()
    console.print(
        f"[green]Reconstructed {len(prompt_lookup)} attack prompts from attack modules.[/green]"
    )
    console.print()

    # ------------------------------------------------------------------
    # Per-model, per-category evaluation
    # ------------------------------------------------------------------
    model_summaries: dict[str, dict] = {}

    # Separate attack results from benign results
    # Benign results would have been stored under a "benign_baseline" attack key
    # if the runner was invoked with benign prompts; otherwise we detect them by metadata.

    for model, attacks in data.get("results", {}).items():
        all_attack_results: list[dict] = []
        all_benign_results: list[dict] = []
        by_category: dict[str, list[dict]] = {}

        for attack_name, results in attacks.items():
            for r in results:
                is_benign = r.get("metadata", {}).get("benign", False)
                if is_benign:
                    all_benign_results.append(r)
                else:
                    all_attack_results.append(r)
                    cat = r.get("category", attack_name)
                    by_category.setdefault(cat, []).append(r)

        # Overall metrics before defense
        orig_asr = _asr(all_attack_results)

        # Overall metrics after defense (with prompt reconstruction for intent checks)
        defended_asr_val, annotated_attacks = _defended_asr(
            all_attack_results, decoder, prompt_lookup
        )

        # False positive rate from stored benign results (if any)
        fpr, annotated_benign = _false_positive_rate(all_benign_results, decoder)

        # Per-category before/after
        cat_stats: dict[str, dict] = {}
        for cat, cat_results in by_category.items():
            orig = _asr(cat_results)
            after, _ = _defended_asr(cat_results, decoder, prompt_lookup)
            defended_count = sum(
                1 for r in cat_results
                if r.get("success") and any(
                    decoder.validate_tool_call(
                        tc.get("name", ""),
                        tc.get("arguments", {}),
                        user_prompt=prompt_lookup.get(r.get("attack_name", "")),
                    )
                    for tc in r.get("tool_calls_made", [])
                )
            )
            cat_stats[cat] = {
                "total": len(cat_results),
                "original_successes": sum(1 for r in cat_results if r.get("success")),
                "original_asr": orig,
                "defended_asr": after,
                "attacks_blocked": defended_count,
                "reduction": orig - after,
            }

        model_summaries[model] = {
            "total_attacks": len(all_attack_results),
            "total_benign": len(all_benign_results),
            "original_asr": orig_asr,
            "defended_asr": defended_asr_val,
            "asr_reduction": orig_asr - defended_asr_val,
            "false_positive_rate": fpr,
            "by_category": cat_stats,
            "annotated_attacks": annotated_attacks,
            "annotated_benign": annotated_benign,
        }

    # ------------------------------------------------------------------
    # Rich output: Model comparison table
    # ------------------------------------------------------------------
    table = Table(
        title="Defense Evaluation: ASR Before vs After ConstrainedDecoder",
        box=box.ROUNDED,
    )
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Original ASR", justify="right")
    table.add_column("Defended ASR", justify="right")
    table.add_column("ASR Reduction", justify="right", style="green")
    table.add_column("FP Rate", justify="right", style="yellow")
    table.add_column("Attacks", justify="right")
    table.add_column("Benign", justify="right")

    for model, stats in model_summaries.items():
        orig = stats["original_asr"]
        defended = stats["defended_asr"]
        reduction = stats["asr_reduction"]
        fpr_val = stats["false_positive_rate"]

        orig_str = f"{orig:.1%}"
        defended_str = f"{defended:.1%}"
        reduction_str = f"-{reduction:.1%}" if reduction > 0 else "0.0%"
        fpr_str = f"{fpr_val:.1%}"

        # Color-code ASR values
        orig_color = "red" if orig > 0.5 else ("yellow" if orig > 0.2 else "green")
        def_color = "red" if defended > 0.5 else ("yellow" if defended > 0.2 else "green")

        table.add_row(
            model,
            f"[{orig_color}]{orig_str}[/{orig_color}]",
            f"[{def_color}]{defended_str}[/{def_color}]",
            f"[green]{reduction_str}[/green]" if reduction > 0 else reduction_str,
            fpr_str,
            str(stats["total_attacks"]),
            str(stats["total_benign"]),
        )

    console.print(table)
    console.print()

    # ------------------------------------------------------------------
    # Per-category breakdown table
    # ------------------------------------------------------------------
    cat_table = Table(
        title="Per-Category ASR Breakdown (Averaged Across Models)",
        box=box.ROUNDED,
    )
    cat_table.add_column("Category", style="magenta")
    cat_table.add_column("Original ASR", justify="right")
    cat_table.add_column("Defended ASR", justify="right")
    cat_table.add_column("Reduction", justify="right", style="green")
    cat_table.add_column("Attacks Blocked", justify="right")

    # Aggregate categories across models
    agg_cats: dict[str, dict] = {}
    for model, stats in model_summaries.items():
        for cat, cstats in stats["by_category"].items():
            if cat not in agg_cats:
                agg_cats[cat] = {
                    "total": 0, "orig_successes": 0, "def_successes": 0, "blocked": 0
                }
            agg_cats[cat]["total"] += cstats["total"]
            agg_cats[cat]["orig_successes"] += cstats["original_successes"]
            agg_cats[cat]["blocked"] += cstats["attacks_blocked"]

    for cat, cdata in sorted(agg_cats.items()):
        total = cdata["total"]
        orig_s = cdata["orig_successes"]
        blocked = cdata["blocked"]
        defended_s = orig_s - blocked
        orig_asr_val = orig_s / total if total else 0.0
        def_asr_val = defended_s / total if total else 0.0
        reduction = orig_asr_val - def_asr_val

        orig_color = "red" if orig_asr_val > 0.5 else ("yellow" if orig_asr_val > 0.2 else "green")
        def_color = "red" if def_asr_val > 0.5 else ("yellow" if def_asr_val > 0.2 else "green")

        cat_table.add_row(
            cat,
            f"[{orig_color}]{orig_asr_val:.1%}[/{orig_color}]",
            f"[{def_color}]{def_asr_val:.1%}[/{def_color}]",
            f"[green]-{reduction:.1%}[/green]" if reduction > 0 else "0.0%",
            str(blocked),
        )

    console.print(cat_table)
    console.print()

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    # Models with attack results (skip models which have 0% baseline ASR)
    vulnerable_models = {m: s for m, s in model_summaries.items() if s["original_asr"] > 0}

    if vulnerable_models:
        avg_orig = sum(s["original_asr"] for s in vulnerable_models.values()) / len(vulnerable_models)
        avg_defended = sum(s["defended_asr"] for s in vulnerable_models.values()) / len(vulnerable_models)
        avg_fpr = sum(s["false_positive_rate"] for s in vulnerable_models.values()) / len(vulnerable_models)

        console.print("[bold]Summary (vulnerable models only):[/bold]")
        console.print(f"  Average original ASR:  [red]{avg_orig:.1%}[/red]")
        console.print(f"  Average defended ASR:  [green]{avg_defended:.1%}[/green]")
        console.print(f"  Average ASR reduction: [green]-{avg_orig - avg_defended:.1%}[/green]")
        console.print(f"  Average false positive rate: [yellow]{avg_fpr:.1%}[/yellow]")
        console.print()

    # All models
    all_orig = sum(s["original_asr"] for s in model_summaries.values()) / len(model_summaries) if model_summaries else 0
    all_def = sum(s["defended_asr"] for s in model_summaries.values()) / len(model_summaries) if model_summaries else 0
    all_fpr = sum(s["false_positive_rate"] for s in model_summaries.values()) / len(model_summaries) if model_summaries else 0
    console.print("[bold]Summary (all models):[/bold]")
    console.print(f"  Average original ASR:  [red]{all_orig:.1%}[/red]")
    console.print(f"  Average defended ASR:  [green]{all_def:.1%}[/green]")
    console.print(f"  Average ASR reduction: [green]-{all_orig - all_def:.1%}[/green]")
    console.print(f"  Average false positive rate: [yellow]{all_fpr:.1%}[/yellow]")
    console.print()

    # ------------------------------------------------------------------
    # Benign prompt simulation (synthetic false positive analysis)
    # ------------------------------------------------------------------
    console.print("[bold]Benign Prompt False Positive Analysis (Simulated)[/bold]")
    console.print(
        "  For each benign prompt, simulates the correct tool call through the decoder."
    )
    console.print(
        "  A false positive = decoder blocks a call that a correct model would make.\n"
    )

    sim_fpr, sim_results = _evaluate_benign_prompts(decoder)

    benign_table = Table(
        title="Benign Prompt Simulation — False Positive Breakdown",
        box=box.ROUNDED,
    )
    benign_table.add_column("Prompt Name", style="cyan")
    benign_table.add_column("Domain", style="magenta")
    benign_table.add_column("Expected Tool", style="blue")
    benign_table.add_column("Blocked?", justify="center")
    benign_table.add_column("Violation Reason")

    fp_list = []
    for r in sim_results:
        blocked = r["falsely_blocked"]
        if blocked:
            fp_list.append(r)
        blocked_str = "[red]YES[/red]" if blocked else "[green]no[/green]"
        violation = r["violations"][0][:80] if r["violations"] else ""
        benign_table.add_row(
            r["prompt_name"],
            r["domain"],
            r["expected_tool"],
            blocked_str,
            violation,
        )

    console.print(benign_table)
    console.print()

    total_benign_sim = len(sim_results)
    fp_count_sim = len(fp_list)

    console.print(f"  Total benign prompts simulated: {total_benign_sim}")
    console.print(f"  Falsely blocked (false positives): [red]{fp_count_sim}[/red]")
    console.print(f"  Simulated false positive rate: [yellow]{sim_fpr:.1%}[/yellow]")
    if fp_list:
        console.print()
        console.print("  [bold]Falsely blocked prompts (policy tuning targets):[/bold]")
        for fp in fp_list:
            console.print(f"    - {fp['prompt_name']} ({fp['domain']}): {fp['expected_tool']}")
            for v in fp["violations"]:
                console.print(f"      Violation: {v[:120]}")
    console.print()

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build clean output (strip large annotated lists for the summary file)
    clean_summary: dict = {}
    for model, stats in model_summaries.items():
        clean_summary[model] = {
            k: v for k, v in stats.items()
            if k not in ("annotated_attacks", "annotated_benign")
        }

    summary_path = output_dir / "defense_summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "source_results": str(results_path),
                "policy": str(policy_path),
                "models": clean_summary,
                "aggregate": {
                    "all_models": {
                        "avg_original_asr": all_orig,
                        "avg_defended_asr": all_def,
                        "avg_asr_reduction": all_orig - all_def,
                        "avg_false_positive_rate": all_fpr,
                    }
                },
                "benign_simulation": {
                    "total_benign_prompts": total_benign_sim,
                    "false_positives": fp_count_sim,
                    "false_positive_rate": sim_fpr,
                    "falsely_blocked": [
                        {
                            "prompt_name": r["prompt_name"],
                            "domain": r["domain"],
                            "expected_tool": r["expected_tool"],
                            "prompt": r["prompt"],
                            "violations": r["violations"],
                        }
                        for r in fp_list
                    ],
                },
            },
            f,
            indent=2,
        )

    # Also save the full annotated results
    annotated_path = output_dir / "annotated_results.json"
    annotated_full: dict = {}
    for model, stats in model_summaries.items():
        annotated_full[model] = {
            "annotated_attacks": stats["annotated_attacks"],
            "annotated_benign": stats["annotated_benign"],
        }
    with open(annotated_path, "w") as f:
        json.dump(annotated_full, f, indent=2)

    # Save benign simulation detail
    benign_sim_path = output_dir / "benign_simulation.json"
    with open(benign_sim_path, "w") as f:
        json.dump(
            {
                "total": total_benign_sim,
                "false_positives": fp_count_sim,
                "false_positive_rate": sim_fpr,
                "results": sim_results,
            },
            f,
            indent=2,
        )

    console.print(f"[green]Saved defense summary to:[/green] {summary_path}")
    console.print(f"[green]Saved annotated results to:[/green] {annotated_path}")
    console.print(f"[green]Saved benign simulation to:[/green] {benign_sim_path}")


if __name__ == "__main__":
    main()
