#!/usr/bin/env bash
# Train a TacitTouch vision-only student on synchronized UniVTAC demonstrations.
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"

task=${TACITTOUCH_TASK:-lift_bottle}
case "$task" in
  lift_bottle|put_bottle_in_shelf|pull_out_key|insert_hole|grasp_classify|insert_HDMI) camera_type=head ;;
  lift_can|insert_tube) camera_type=all ;;
  *) echo "Unsupported UniVTAC task: $task" >&2; exit 2 ;;
esac

num_gpus=${SIMPLETOUCH_NUM_GPUS:-8}
if ! [[ "$num_gpus" =~ ^[1-9][0-9]*$ ]] || (( num_gpus > 64 || 64 % num_gpus != 0 )); then
  echo "SIMPLETOUCH_NUM_GPUS must divide the global batch of 64" >&2
  exit 2
fi
export SIMPLETAC_GPU_PLACEMENT=${SIMPLETOUCH_GPU_PLACEMENT:-0-$((num_gpus - 1))}
export EMBODIED_PATH="$repo_root/examples/sft"
export PYTHONPATH="$repo_root:$repo_root/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"

: "${SIMPLETOUCH_DATA_ROOT:?Set the UniVTAC dataset root}"
: "${SIMPLETOUCH_NORM_STATS_PATH:?Set the task normalization stats}"
: "${SIMPLETOUCH_BASE_MODEL_PATH:?Set the converted pi0.5 base model directory}"
: "${SIMPLETOUCH_T3_PATH:?Set the T3-large encoder directory}"
: "${SIMPLETOUCH_TOKENIZER_PATH:?Set the PaliGemma tokenizer path}"
: "${TACITTOUCH_TEACHER_CHECKPOINT:?Set the completed SimpleTouch checkpoint directory}"

exec "${SIMPLETOUCH_PYTHON:-python}" "$repo_root/examples/sft/train_vla_sft.py" \
  --config-name simpletac_univtac '+tacittouch=univtac' \
  "data.task=$task" "data.camera_type=$camera_type" \
  "actor.micro_batch_size=$((64 / num_gpus))" "$@"
