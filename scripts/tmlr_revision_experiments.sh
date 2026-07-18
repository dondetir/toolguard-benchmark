#!/usr/bin/env bash
# TMLR revision experiments (Reviewer jiww): temperature sweep + small-model size gradient.
# Sequential (one model in VRAM at a time). Writes a per-run results.json + a global sentinel.
set -u
cd "$(dirname "$0")/.."
ROOT="experiments/tmlr-revision"
LOG="$ROOT/run.log"
mkdir -p "$ROOT"
: > "$LOG"
# Clear stale sentinels so completion detection can't misfire on a re-run.
rm -f "$ROOT/.all_complete" "$ROOT/.failures"
find "$ROOT" -name .complete -delete 2>/dev/null || true

run() {  # args: label, output_subdir, extra flags...
  local label="$1"; shift
  local out="$1"; shift
  echo "=== [$(date -u +%H:%M:%S)] START $label -> $out ===" | tee -a "$LOG"
  if python3 scripts/run_redteam.py --attack all --output "$out" "$@" >>"$LOG" 2>&1; then
    touch "$out/.complete"
    echo "=== [$(date -u +%H:%M:%S)] DONE  $label ===" | tee -a "$LOG"
  else
    echo "=== [$(date -u +%H:%M:%S)] FAIL  $label (see $LOG) ===" | tee -a "$LOG"
    echo "$label" >> "$ROOT/.failures"
  fi
}

# --- Experiment B: within-family size gradient (T=0.7, core 50 x 10) ---
run "gradient qwen2.5:0.5b" "$ROOT/small-models/qwen2.5-0.5b" --model qwen2.5:0.5b --runs 10
run "gradient qwen2.5:1.5b" "$ROOT/small-models/qwen2.5-1.5b" --model qwen2.5:1.5b --runs 10

# --- Experiment A: temperature sweep on a capable model (qwen2.5:3b, paper T=0.7=52%) ---
# T=0.7 is included as a same-harness reproduction check: it must land near the paper's 52%,
# and it makes the 0.0/0.3/0.7/1.0 curve self-consistent from one harness state.
run "temp T=0.0" "$ROOT/temp-sweep/T0.0" --model qwen2.5:3b --runs 3  --temperature 0.0
run "temp T=0.3" "$ROOT/temp-sweep/T0.3" --model qwen2.5:3b --runs 10 --temperature 0.3
run "temp T=0.7" "$ROOT/temp-sweep/T0.7" --model qwen2.5:3b --runs 10 --temperature 0.7
run "temp T=1.0" "$ROOT/temp-sweep/T1.0" --model qwen2.5:3b --runs 10 --temperature 1.0

if [ ! -f "$ROOT/.failures" ]; then
  touch "$ROOT/.all_complete"
  echo "=== [$(date -u +%H:%M:%S)] ALL COMPLETE ===" | tee -a "$LOG"
else
  echo "=== [$(date -u +%H:%M:%S)] COMPLETE WITH FAILURES: $(cat $ROOT/.failures | tr '\n' ' ') ===" | tee -a "$LOG"
fi
