#!/usr/bin/env python3
"""Analyze results from ToolGuard red-teaming experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.table import Table

from src.evaluation.benchmark import load_experiment_results, run_benchmark, save_benchmark_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ToolGuard Results Analyzer")
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        help="Path to experiment directory containing results.json",
    )
    parser.add_argument(
        "--save-report",
        type=str,
        default=None,
        help="Save benchmark report to this path (JSON)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()

    experiment_dir = Path(args.experiment)
    if not (experiment_dir / "results.json").exists():
        console.print(f"[red]No results.json found in {experiment_dir}[/red]")
        sys.exit(1)

    console.print(f"[bold]Analyzing: {experiment_dir}[/bold]\n")

    report = run_benchmark(str(experiment_dir))

    # Model comparison table
    table = Table(title="Model Comparison")
    table.add_column("Model", style="cyan")
    table.add_column("ASR", justify="right")
    table.add_column("Safety-Utility", justify="right")

    for model, metrics in report["model_comparison"].items():
        asr = metrics["asr"]
        su = metrics["safety_utility"]
        asr_style = "red" if asr > 0.5 else ("yellow" if asr > 0.2 else "green")
        su_style = "green" if su > 0.7 else ("yellow" if su > 0.5 else "red")
        table.add_row(
            model,
            f"[{asr_style}]{asr:.1%}[/{asr_style}]",
            f"[{su_style}]{su:.3f}[/{su_style}]",
        )
    console.print(table)

    # Per-model detail
    for model, detail in report["per_model_detail"].items():
        console.print(f"\n[bold cyan]{model}[/bold cyan] — {detail['total_prompts']} prompts")

        cat_table = Table(title="By Category")
        cat_table.add_column("Category")
        cat_table.add_column("Total", justify="right")
        cat_table.add_column("Succeeded", justify="right")
        cat_table.add_column("ASR", justify="right")
        for cat, stats in detail["by_category"].items():
            asr = stats["asr"]
            style = "red" if asr > 0.5 else ("yellow" if asr > 0.2 else "green")
            cat_table.add_row(
                cat, str(stats["total"]), str(stats["succeeded"]),
                f"[{style}]{asr:.1%}[/{style}]",
            )
        console.print(cat_table)

        sev_table = Table(title="By Severity")
        sev_table.add_column("Severity")
        sev_table.add_column("Total", justify="right")
        sev_table.add_column("Succeeded", justify="right")
        sev_table.add_column("ASR", justify="right")
        for sev, stats in detail["by_severity"].items():
            asr = stats["asr"]
            style = "red" if asr > 0.5 else ("yellow" if asr > 0.2 else "green")
            sev_table.add_row(
                sev, str(stats["total"]), str(stats["succeeded"]),
                f"[{style}]{asr:.1%}[/{style}]",
            )
        console.print(sev_table)

    if args.save_report:
        save_benchmark_report(report, args.save_report)
        console.print(f"\nReport saved to: {args.save_report}")


if __name__ == "__main__":
    main()
