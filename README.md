# ToolGuard: Red-Teaming Small Language Model Tool Calling

Code and benchmark for the TMLR paper *ToolGuard: Red-Teaming Small Language Model Tool Calling on
Consumer Hardware* (Ravitez Dondeti, 2026). Benchmark and defense code for studying adversarial
robustness of sub-4B language models when they call tools.

Contents: an adversarial benchmark (5 attack categories over simulated tool schemas), a red-team harness
that runs it against local models via [Ollama](https://ollama.com), and `ToolGuard`, a WAF-inspired
post-hoc policy layer (`ConstrainedDecoder`) evaluated offline on recorded tool calls.

## What is here

- **Benchmark: 153 adversarial prompts** across 7 domains.
  - **50 core prompts** in 5 categories: parameter_injection (13), tool_substitution (8),
    privilege_escalation (10), data_exfiltration (10), chain_attacks (9), in `src/attacks/`.
  - **103 expanded prompts** (`src/attacks/expanded_suite.py`), 103 strictly held out from policy
    development; 68 are policy-evading by design.
- **41 benign prompts** (`src/attacks/benign_baseline.py`) for false-positive measurement.
- **25 tool schemas** over 7 domains (banking, filesystem, email, system, database, calendar, cloud/DevOps)
  in `src/harness/tool_schemas.py`. 16 are the core-domain tools.
- **Both policy files**: `configs/defense_policy.yaml` (full) and `configs/defense_policy_train.yaml`
  (train-only, for leakage-free held-out evaluation).
- **Multi-turn scaffold** (`src/attacks/multi_turn.py`) for the cross-call / indirect-injection setting.
- **Raw results** backing the revision tables (temperature sweep, size ladder, four newer models) under
  `experiments/tmlr-revision/`.

## Install

```bash
pip install -e .            # Python >= 3.10; deps: pyyaml, httpx, rich
ollama serve               # separate terminal; pull the models you want to test, e.g.:
ollama pull qwen3:4b-instruct
```

## Reproduce

```bash
# 1. Red-team a model (50 core prompts x 10 runs = 500 runs). Raw outputs -> results.json.
python scripts/run_redteam.py --model qwen3:4b-instruct --attack all --runs 10 \
    --output experiments/my-run

# 2. Summarize ASR by category / severity.
python scripts/analyze_results.py --experiment experiments/my-run

# 3. Offline defense eval (full policy): ASR before/after + simulated FPR. No model needed.
python scripts/evaluate_defense.py --results experiments/my-run/results.json \
    --policy configs/defense_policy.yaml

# 4. Leakage-free held-out eval (train-only policy).
python scripts/evaluate_holdout.py --results experiments/my-run/results.json \
    --policy configs/defense_policy_train.yaml

# Temperature sweep + within-family size ladder (revision experiments).
bash scripts/tmlr_revision_experiments.sh

# Optional: pin a decoding temperature.
python scripts/run_redteam.py --model qwen2.5:3b --attack all --runs 10 --temperature 0.0

# tests
pytest -q
```

## Model configuration (`configs/models.yaml`)

- `tool_format: json` — native Ollama tool-calling (most models).
- `think: false` — disables reasoning on hybrid models (e.g. `qwen3.5:4b`) so tool-calling latency is usable.
- `tool_format: prompted` — for models with no native tool API (e.g. `gemma3:4b`): schemas are injected as a
  system prompt and `<tool_call>` blocks are parsed from the response.

## Layout

```
src/attacks/      one module per attack category (+ benign_baseline, expanded_suite, multi_turn scaffold)
src/harness/      Ollama/HF clients, 25 tool schemas, experiment runner, tool-call recovery
src/defenses/     ConstrainedDecoder policy layer
src/evaluation/   ASR / metrics / benchmark report
scripts/          run_redteam / analyze_results / evaluate_defense / evaluate_holdout / audits
configs/          models.yaml, attacks.yaml, defense_policy.yaml, defense_policy_train.yaml
experiments/      tmlr-revision/ raw results for the revision tables
tests/            unit tests (pytest)
```

Hardware note: designed to run on consumer hardware without a discrete GPU (e.g. an AMD 780M iGPU via
Ollama). Generation is memory-bandwidth-bound; defense evaluation is offline and instant.

## License

MIT (see `LICENSE`).
