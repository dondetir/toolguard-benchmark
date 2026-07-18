#!/usr/bin/env python3
"""Entry point for ToolGuard red-teaming experiments."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from src.harness.runner import ExperimentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ToolGuard Red-Teaming Runner")
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Model name (e.g., 'qwen3.5:2b') or 'all'",
    )
    parser.add_argument(
        "--attack",
        type=str,
        default="all",
        help="Attack category (e.g., 'parameter_injection') or 'all'",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per attack prompt (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: experiments/YYYYMMDD-HHMMSS/)",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434",
        help="Ollama API URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--defense",
        type=str,
        default=None,
        help="Path to defense policy YAML (e.g., configs/defense_policy.yaml). "
             "When set, the ConstrainedDecoder validates each generated tool call.",
    )
    parser.add_argument(
        "--expanded",
        action="store_true",
        help="Also run the expanded held-out suite (TMLR revision): +103 additive prompts "
             "across 5 categories and 7 domains (incl. calendar + cloud).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override sampling temperature for all runs (default: None = use "
             "configs/attacks.yaml value, 0.7). Used for the TMLR temperature-sensitivity sweep.",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    console = Console()

    console.print("[bold]ToolGuard Red-Teaming Runner[/bold]")
    console.print(f"  Model:   {args.model}")
    console.print(f"  Attack:  {args.attack}")
    console.print(f"  Runs:    {args.runs}")
    console.print(f"  Temp:    {args.temperature if args.temperature is not None else 'config (0.7)'}")
    console.print(f"  Defense: {args.defense or 'None'}")
    console.print()

    runner = ExperimentRunner(
        output_dir=args.output,
        ollama_url=args.ollama_url,
        defense_policy=args.defense,
        include_expanded=args.expanded,
        temperature_override=args.temperature,
    )

    # Check Ollama availability
    if not await runner.client.is_available():
        console.print("[red]Error: Ollama is not running.[/red]")
        console.print(f"  Expected at: {args.ollama_url}")
        console.print("  Start with: ollama serve")
        sys.exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Running experiments...", total=None)

        try:
            results = await runner.run(
                model_filter=args.model,
                attack_filter=args.attack,
                runs=args.runs,
            )
        except Exception as e:
            import traceback
            console.print(f"[red]Error: {e}[/red]")
            console.print(f"[red]{traceback.format_exc()}[/red]")
            sys.exit(1)

        progress.update(task, completed=True, total=1)

    # Print summary table
    summary = ExperimentRunner.compute_summary(results)
    console.print()

    table = Table(title="Red-Teaming Results Summary")
    table.add_column("Model", style="cyan")
    table.add_column("Attack Category", style="magenta")
    table.add_column("Total", justify="right")
    table.add_column("Succeeded", justify="right")
    table.add_column("ASR", justify="right", style="red")

    for model, attacks in summary.items():
        for attack_name, stats in attacks.items():
            asr_str = f"{stats['asr']:.1%}"
            style = "red" if stats["asr"] > 0.5 else ("yellow" if stats["asr"] > 0.2 else "green")
            table.add_row(
                model,
                attack_name,
                str(stats["total"]),
                str(stats["succeeded"]),
                f"[{style}]{asr_str}[/{style}]",
            )

    console.print(table)
    console.print(f"\nResults saved to: {runner.output_dir / 'results.json'}")


if __name__ == "__main__":
    asyncio.run(main())
