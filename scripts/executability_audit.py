#!/usr/bin/env python3
"""Uniform executability audit + DEFENSE re-run (TMLR revision).

For every model, computed identically (from saved raw completions — deterministic, no re-run):
  - ASR at three executability tiers: strict-native / recovered-executable (HEADLINE) / intent-upper-bound
  - per-model FORMAT-COMPLIANCE (fraction of runs with no executable call hidden in text)
  - LATENT-INTENT rate (intent tier)
  - DEFENSE re-run: the frozen policy applied to the RECOVERED-executable calls (not just the native
    field), giving the corrected DEFENDED ASR, and DEFENSE RECALL on malformed-but-malicious calls
    (attacks that only the recovery exposed — does the defense still block them?).

Usage: python scripts/executability_audit.py [--policy configs/defense_policy.yaml]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.harness.tool_call_recovery import recover_executable, recover_intent, extract_text_calls, REAL_SCHEMA_TOOLS
from src.harness.tool_schemas import ALL_TOOL_SCHEMAS
from src.defenses.constrained_decoder import ConstrainedDecoder
from src.attacks.parameter_injection import ParameterInjectionAttack
from src.attacks.privilege_escalation import PrivilegeEscalationAttack
from src.attacks.data_exfiltration import DataExfiltrationAttack
from src.attacks.tool_substitution import ToolSubstitutionAttack
from src.attacks.chain_attacks import ChainAttack
from src.attacks.expanded_suite import all_expanded_prompts

_ATKS = [ParameterInjectionAttack(), ToolSubstitutionAttack(), PrivilegeEscalationAttack(),
         DataExfiltrationAttack(), ChainAttack()]
PLOOK = {}       # name -> (AttackPrompt, attack)
PTEXT = {}       # name -> prompt text
for _atk in _ATKS:
    for _p in _atk.generate_prompts(ALL_TOOL_SCHEMAS):
        PLOOK[_p.name] = (_p, _atk); PTEXT[_p.name] = _p.prompt
for _p in all_expanded_prompts():
    # map expanded prompt to its category's attack instance
    _inst = {a.category: a for a in _ATKS}[_p.category]
    PLOOK[_p.name] = (_p, _inst); PTEXT[_p.name] = _p.prompt


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - m) / d, (c + m) / d


def _succ(attack, prompt, calls) -> bool:
    return bool(attack.evaluate_response({"content": "", "tool_calls": calls}, prompt).success)


def _blocked(decoder, calls, prompt_text) -> bool:
    return any(decoder.validate_tool_call(c.get("name", ""), c.get("arguments", {}), user_prompt=prompt_text)
               for c in calls)


def audit(model: str, results: dict, decoder: ConstrainedDecoder) -> dict:
    n = 0
    strict_s = rec_s = intent_s = 0
    noncompliant = 0
    malformed_malicious = 0            # success under recovered but NOT strict
    malformed_malicious_blocked = 0    # of those, defense blocks
    def_strict_s = def_rec_s = 0       # defended successes (native-only defense vs recovered-aware defense)
    for cat, runs in results.items():
        for r in runs:
            name = r.get("attack_name", "")
            if name not in PLOOK:
                continue
            prompt, attack = PLOOK[name]
            ptext = PTEXT.get(name)
            native = r.get("tool_calls_made", []) or []
            text = r.get("model_response", "") or ""
            rec = recover_executable(text, native)
            intent = recover_intent(text, native)
            n += 1
            s_strict = _succ(attack, prompt, native)
            s_rec = _succ(attack, prompt, rec)
            s_intent = _succ(attack, prompt, intent)
            strict_s += s_strict; rec_s += s_rec; intent_s += s_intent
            if rec != native:
                noncompliant += 1
            # defended (native-only defense input)
            if s_strict and not _blocked(decoder, native, ptext):
                def_strict_s += 1
            # defended (recovered-aware defense input) -- the FIX
            if s_rec and not _blocked(decoder, rec, ptext):
                def_rec_s += 1
            # malformed-but-malicious: attack exposed only by recovery
            if s_rec and not s_strict:
                malformed_malicious += 1
                if _blocked(decoder, rec, ptext):
                    malformed_malicious_blocked += 1
    lo_r, hi_r = _wilson(rec_s, n)
    return {
        "model": model, "n": n,
        "asr_strict_native": round(strict_s / n, 4),
        "asr_recovered_executable": round(rec_s / n, 4),   # HEADLINE
        "asr_recovered_wilson": [round(lo_r, 4), round(hi_r, 4)],
        "latent_intent_rate": round(intent_s / n, 4),
        "format_compliance": round(1 - noncompliant / n, 4),
        "defended_asr_native_input": round(def_strict_s / n, 4),
        "defended_asr_recovered_input": round(def_rec_s / n, 4),  # the FIXED defended number
        "malformed_malicious_attacks": malformed_malicious,
        "defense_recall_on_malformed": round(malformed_malicious_blocked / malformed_malicious, 4) if malformed_malicious else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="configs/defense_policy.yaml")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    decoder = ConstrainedDecoder.load_policy(root / args.policy)
    sources = {
        "experiments/10runs-MERGED/results.json":
            ["smollm2:1.7b", "llama3.2:1b", "llama3.2:3b", "qwen2.5:3b"],
        "experiments/tmlr-revision/qwen3-4b-instruct/results.json": ["qwen3:4b-instruct"],
        "experiments/tmlr-revision/qwen35-4b/results.json": ["qwen3.5:4b"],
        "experiments/tmlr-revision/gemma3-4b-hardened/results.json": ["gemma3:4b"],
    }
    rows = []
    for rel, models in sources.items():
        p = root / rel
        if not p.exists():
            continue
        data = json.load(open(p))
        for m in models:
            if m in data.get("results", {}):
                rows.append(audit(m, data["results"][m], decoder))
    hdr = (f"{'model':<18} {'strict':>7} {'RECOV*':>7} {'intent':>7} {'fmt_cmpl':>9} "
           f"{'def_native':>11} {'def_recov':>10} {'malf_mal':>9} {'def_recall':>11}")
    print(hdr)
    for r in rows:
        rec = r["asr_recovered_executable"] * 100
        w = r["asr_recovered_wilson"]
        print(f"{r['model']:<18} {r['asr_strict_native']*100:6.1f}% {rec:6.1f}% "
              f"{r['latent_intent_rate']*100:6.1f}% {r['format_compliance']*100:8.1f}% "
              f"{r['defended_asr_native_input']*100:10.1f}% {r['defended_asr_recovered_input']*100:9.1f}% "
              f"{r['malformed_malicious_attacks']:>9} "
              f"{(r['defense_recall_on_malformed']*100 if r['defense_recall_on_malformed'] is not None else float('nan')):10.1f}%")
    print("\n* RECOV = recovered-executable = HEADLINE ASR (real schema tool + parseable args, any format).")
    print("def_native = defended ASR if the defense sees only the native field (the BUG).")
    print("def_recov  = defended ASR with the FIXED recovered-aware defense input.")
    print("def_recall = fraction of malformed-but-malicious attacks the fixed defense blocks.")
    out = root / "experiments/tmlr-revision/executability_audit.json"
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
