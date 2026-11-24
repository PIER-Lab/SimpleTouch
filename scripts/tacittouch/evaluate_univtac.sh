#!/usr/bin/env bash
# Use the same UniVTAC evaluator with the student-only observation contract.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
: "${TACITTOUCH_TASK:?Set the UniVTAC task}"
: "${TACITTOUCH_CHECKPOINT:?Set the trained TacitTouch checkpoint directory}"

export UNIVTAC_ENVIRONMENT=tacittouch
exec "${SIMPLETOUCH_PYTHON:-python}" "$repo_root/scripts/univtac/eval_simpletac_managed.py" \
  --checkpoint "$TACITTOUCH_CHECKPOINT" --task "$TACITTOUCH_TASK" \
  --gpus "${SIMPLETOUCH_EVAL_GPUS:-${GPU_ID:-0}}" "$@" --use-tactile false
