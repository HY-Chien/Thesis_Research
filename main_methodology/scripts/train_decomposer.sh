#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_LLAMA_FACTORY_DIR="$(cd "$PROJECT_DIR/../.." && pwd)/LLaMA-Factory"

LLAMA_FACTORY_DIR="${LLAMA_FACTORY_DIR:-$DEFAULT_LLAMA_FACTORY_DIR}"
SEEK_DATA_DIR="${SEEK_DATA_DIR:-/tmp/SEEK-decomposer-12k}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LLAMAFACTORY_CLI="${LLAMAFACTORY_CLI:-llamafactory-cli}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES
CONFIG_PATH="$PROJECT_DIR/configs/llama3_8b_decomposer_lora_3ds.yaml"
EXPECTED_REVISION="7e20db5735443861fad91766614b3415ff8966a4"
MODE="train"

if [[ "${1:-}" == "--check-only" ]]; then
  MODE="check"
elif [[ "${1:-}" == "--smoke-test" ]]; then
  MODE="smoke"
elif [[ $# -ne 0 ]]; then
  echo "Usage: bash scripts/train_decomposer.sh [--check-only|--smoke-test]" >&2
  exit 2
fi

"$PYTHON_BIN" -c \
  'import sys; assert sys.version_info >= (3, 11), "Python 3.11 or newer is required"'

"$PYTHON_BIN" -c '
import torch
import torchaudio
import torchvision

expected = {
    "torch": (torch.__version__, "2.10.0"),
    "torchvision": (torchvision.__version__, "0.25.0"),
    "torchaudio": (torchaudio.__version__, "2.10.0"),
}
for name, (actual, wanted) in expected.items():
    if actual.split("+")[0] != wanted:
        raise SystemExit(f"{name}=={wanted} is required; found {actual}")
'

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Training configuration not found: $CONFIG_PATH" >&2
  exit 2
fi

if [[ ! -d "$LLAMA_FACTORY_DIR/.git" ]]; then
  echo "LLaMA-Factory checkout not found: $LLAMA_FACTORY_DIR" >&2
  echo "Set LLAMA_FACTORY_DIR to the pinned checkout described in README.md." >&2
  exit 2
fi

ACTUAL_REVISION="$(git -C "$LLAMA_FACTORY_DIR" rev-parse HEAD)"
if [[ "$ACTUAL_REVISION" != "$EXPECTED_REVISION" ]]; then
  echo "Unexpected LLaMA-Factory revision: $ACTUAL_REVISION" >&2
  echo "Expected: $EXPECTED_REVISION" >&2
  exit 2
fi

if ! command -v "$LLAMAFACTORY_CLI" >/dev/null 2>&1; then
  echo "llamafactory-cli was not found in the active environment." >&2
  echo "Install the pinned checkout with: python -m pip install -e \"$LLAMA_FACTORY_DIR\"" >&2
  exit 2
fi

"$PYTHON_BIN" "$SCRIPT_DIR/prepare_llamafactory_data.py" \
  --llamafactory-dir "$LLAMA_FACTORY_DIR" \
  --dataset-dir "$SEEK_DATA_DIR" \
  --config "$CONFIG_PATH"

"$LLAMAFACTORY_CLI" version >/dev/null

if [[ "$MODE" == "check" ]]; then
  echo "Training preflight passed."
  echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
  echo "Config: $CONFIG_PATH"
  echo "Output: $LLAMA_FACTORY_DIR/saves/seek/llama3-8b-decomposer-musique-2wiki-popqa"
  exit 0
fi

"$PYTHON_BIN" -c \
  'import torch; assert torch.cuda.is_available(), "A CUDA GPU is required for training"'

cd "$LLAMA_FACTORY_DIR"

if [[ "$MODE" == "smoke" ]]; then
  SMOKE_OUTPUT_DIR="$(mktemp -d /tmp/seek-decomposer-smoke.XXXXXX)"
  "$LLAMAFACTORY_CLI" train "$CONFIG_PATH" \
    max_steps=1 \
    num_train_epochs=1.0 \
    logging_steps=1 \
    save_strategy=steps \
    save_steps=999999 \
    plot_loss=false \
    output_dir="$SMOKE_OUTPUT_DIR"
  echo "One-step training smoke test passed: $SMOKE_OUTPUT_DIR"
  exit 0
fi

exec "$LLAMAFACTORY_CLI" train "$CONFIG_PATH"
