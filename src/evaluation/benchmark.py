"""Benchmark runner for standardized evaluation across models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.evaluation.metrics import (
    aggregate_model_comparison,
    attack_success_rate,
    per_category_breakdown,
    per_severity_breakdown,
    safety_utility_ratio,
)


def load_experiment_results(experiment_dir: str) -> dict[str, Any]:
    """Load results.json from an experiment directory."""
    path = Path(experiment_dir) / "results.json"
    with open(path) as f:
        return json.load(f)


def flatten_results(data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Flatten nested results into {model: [result_dicts]}."""
    flat: dict[str, list[dict[str, Any]]] = {}
    for model, attacks in data.get("results", {}).items():
        flat[model] = []
        for _attack_name, results in attacks.items():
            flat[model].extend(results)
    return flat


def run_benchmark(experiment_dir: str) -> dict[str, Any]:
    """Run full benchmark analysis on an experiment directory.

    Returns a structured report with per-model and per-category metrics.
    """
    data = load_experiment_results(experiment_dir)
    flat = flatten_results(data)

    report: dict[str, Any] = {
        "experiment": {
            "timestamp": data.get("timestamp"),
            "models": data.get("models"),
            "attacks": data.get("attacks"),
            "runs_per_attack": data.get("runs_per_attack"),
        },
        "model_comparison": aggregate_model_comparison(flat),
        "per_model_detail": {},
    }

    for model, results in flat.items():
        asr = attack_success_rate(results)
        report["per_model_detail"][model] = {
            "overall_asr": asr,
            "safety_utility": safety_utility_ratio(asr, 1.0 - asr),
            "by_category": per_category_breakdown(results),
            "by_severity": per_severity_breakdown(results),
            "total_prompts": len(results),
        }

    return report


def save_benchmark_report(report: dict[str, Any], output_path: str) -> None:
    """Save benchmark report as JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
