#!/usr/bin/env python3
"""Prompt-level analysis for the expanded benchmark (TMLR revision).

Runs within a prompt are NOT independent (shared prompt/model/system-prompt), so we do NOT put
Wilson CIs on the run count. The statistical unit is the PROMPT:
  - per-prompt ASR = successes / runs_per_prompt
  - point estimate = mean of per-prompt ASR across prompts
  - 95% CI = CLUSTER BOOTTRAP resampling PROMPTS (not runs) with replacement
  - ICC(1) reported to justify runs-per-prompt (one-way random-effects on the binary outcome)

Also computes the DEFENDED per-prompt ASR (frozen train policy applied offline) and splits everything by
provenance so the "generalization" claim is honest and not tautological:
  - original (50) vs expanded (103) prompts
  - original 5 domains vs new 2 domains (calendar/cloud)
  - policy_evading_by_design True vs False

Usage: python scripts/expanded_analysis.py --results <dir>/results.json [--policy configs/defense_policy_train.yaml]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.defenses.constrained_decoder import ConstrainedDecoder
from src.harness.tool_schemas import ALL_TOOL_SCHEMAS
from src.attacks.parameter_injection import ParameterInjectionAttack
from src.attacks.privilege_escalation import PrivilegeEscalationAttack
from src.attacks.data_exfiltration import DataExfiltrationAttack
from src.attacks.tool_substitution import ToolSubstitutionAttack
from src.attacks.chain_attacks import ChainAttack
from src.attacks.expanded_suite import all_expanded_prompts

random.seed(42)


def _prompt_lookup():
    """name -> (AttackPrompt, attack_instance, prompt_text). Used to RE-SCORE via the uniform
    would-dispatch rule (recover_executable) rather than trusting the stored native-field success."""
    atks = [ParameterInjectionAttack(), ToolSubstitutionAttack(), PrivilegeEscalationAttack(),
            DataExfiltrationAttack(), ChainAttack()]
    by_cat = {a.category: a for a in atks}
    lut = {}
    for atk in atks:
        for p in atk.generate_prompts(ALL_TOOL_SCHEMAS):
            lut[p.name] = (p, atk, p.prompt)
    for p in all_expanded_prompts():
        lut[p.name] = (p, by_cat[p.category], p.prompt)
    return lut


def _per_prompt(results: dict, decoder: ConstrainedDecoder | None, plut: dict) -> dict:
    """Group runs by prompt -> {meta, n, base, defn}, scored under the UNIFORM would-dispatch rule.

    Success and defense are recomputed from the RECOVERED-executable calls (native ∪ real-tool text
    calls), NOT the stored native-field success — otherwise <100%-compliant models (e.g. llama3.2:3b)
    would silently inherit the native-field blind spot in the headline table.
    """
    from src.harness.tool_call_recovery import recover_executable
    by = {}
    for cat, runs in results.items():
        for r in runs:
            name = r.get("attack_name", "")
            if name not in plut:
                continue
            prompt, attack, ptext = plut[name]
            e = by.setdefault(name, {"n": 0, "base": 0, "defn": 0, "meta": r.get("metadata", {})})
            e["n"] += 1
            native = r.get("tool_calls_made", []) or []
            calls = recover_executable(r.get("model_response", "") or "", native)
            success = bool(attack.evaluate_response({"content": "", "tool_calls": calls}, prompt).success)
            e["base"] += 1 if success else 0
            if decoder is not None:
                defended_success = success
                if success and calls:
                    blocked = any(
                        decoder.validate_tool_call(tc.get("name", ""), tc.get("arguments", {}),
                                                   user_prompt=ptext)
                        for tc in calls)
                    if blocked:
                        defended_success = False
                e["defn"] += 1 if defended_success else 0
    return by


def _cluster_bootstrap(prompt_asrs: list[float], B: int = 10000) -> tuple[float, float, float]:
    if not prompt_asrs:
        return 0.0, 0.0, 0.0
    n = len(prompt_asrs)
    mean = sum(prompt_asrs) / n
    means = []
    for _ in range(B):
        s = [prompt_asrs[random.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    return mean, means[int(0.025 * B)], means[int(0.975 * B)]


def _icc1(per: dict, key: str) -> float:
    """One-way random-effects ICC(1) on the binary outcome, runs nested in prompts."""
    groups = [(e["n"], e[key]) for e in per.values() if e["n"] > 0]
    N = sum(n for n, _ in groups)
    k = N / len(groups)  # avg runs per prompt
    grand = sum(s for _, s in groups) / N
    ss_between = sum(n * ((s / n) - grand) ** 2 for n, s in groups)
    # within-group SS for binary: sum over group of [s*(1-p)^2 + (n-s)*p^2] = n*p*(1-p)
    ss_within = sum(n * (s / n) * (1 - s / n) for n, s in groups)
    df_b = len(groups) - 1
    df_w = N - len(groups)
    msb = ss_between / df_b if df_b else 0.0
    msw = ss_within / df_w if df_w else 0.0
    denom = msb + (k - 1) * msw
    return (msb - msw) / denom if denom else 0.0


def _summarize(per: dict, subset_filter, label: str, has_def: bool):
    names = [nm for nm, e in per.items() if subset_filter(e["meta"], nm)]
    base = [per[nm]["base"] / per[nm]["n"] for nm in names]
    bm, blo, bhi = _cluster_bootstrap(base)
    line = (f"{label:<34} n_prompts={len(names):>3}  baseASR={bm*100:5.1f}% "
            f"[{blo*100:4.1f},{bhi*100:4.1f}]")
    if has_def:
        dfa = [per[nm]["defn"] / per[nm]["n"] for nm in names]
        dm, dlo, dhi = _cluster_bootstrap(dfa)
        line += f"  defASR={dm*100:5.1f}% [{dlo*100:4.1f},{dhi*100:4.1f}]  reduction={ (bm-dm)*100:5.1f}pp"
    print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--policy", default="configs/defense_policy_train.yaml")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    data = json.load(open(root / args.results))
    plut = _prompt_lookup()
    decoder = ConstrainedDecoder.load_policy(root / args.policy) if args.policy else None
    has_def = decoder is not None

    for model, results in data.get("results", {}).items():
        per = _per_prompt(results, decoder, plut)
        print(f"\n===== {model}  (unit=prompt, cluster-bootstrap over prompts, seed 42) =====")
        _summarize(per, lambda m, n: True, "ALL", has_def)
        _summarize(per, lambda m, n: m.get("source") != "expanded", "original-50", has_def)
        _summarize(per, lambda m, n: m.get("source") == "expanded", "expanded-103 (held-out)", has_def)
        _summarize(per, lambda m, n: m.get("source") == "expanded" and m.get("domain") in ("calendar", "cloud"),
                   "  expanded: NEW domains", has_def)
        _summarize(per, lambda m, n: m.get("source") == "expanded" and m.get("domain") not in ("calendar", "cloud"),
                   "  expanded: original domains", has_def)
        _summarize(per, lambda m, n: m.get("source") == "expanded" and m.get("policy_evading_by_design") is True,
                   "  expanded: policy-EVADING", has_def)
        _summarize(per, lambda m, n: m.get("source") == "expanded" and m.get("policy_evading_by_design") is False,
                   "  expanded: policy-COVERED", has_def)
        print(f"  ICC(1) baseline = {_icc1(per, 'base'):.3f}  (runs within prompt; justifies runs/prompt)")


if __name__ == "__main__":
    main()
