#!/bin/bash
# Run Qwen3.5 red-teaming sweep via ROCm container with HF transformers.
# Uses native tool-calling format (apply_chat_template with tools).
#
# WARNING: Run ONE sweep at a time. Concurrent sweeps compete for the AMD 780M iGPU
# (ROCm does not support multi-process GPU access on gfx1103) and produce empty results.
# If another sweep is already running, this script will exit immediately.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Guard against concurrent GPU usage: fail fast if another rocm container is running
if docker ps --filter "ancestor=rocm-gfx1103-base:latest" --format '{{.ID}}' | grep -q .; then
  echo "ERROR: Another rocm-gfx1103-base container is already running." >&2
  echo "       Wait for it to finish before starting a new sweep." >&2
  docker ps --filter "ancestor=rocm-gfx1103-base:latest" >&2
  exit 1
fi

RUNS="${1:-3}"

for model in Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B; do
  echo "=== Testing $model ==="
  docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render \
    --ipc=host --shm-size=8g \
    --security-opt seccomp=unconfined \
    -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
    -e PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512 \
    -e GPU_MAX_ALLOC_PERCENT=90 \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -v "$SCRIPT_DIR:/workspace" \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -w /workspace \
    rocm-gfx1103-base:latest \
    python scripts/run_redteam_hf.py \
      --model "$model" \
      --attack all \
      --runs "$RUNS" \
      --max-tokens 256 \
      --output "experiments/qwen35-$(date +%Y%m%d-%H%M%S)"
  echo ""
done

echo "=== Sweep complete ==="
