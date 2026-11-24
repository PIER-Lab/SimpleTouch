# TacitTouch on UniVTAC

[TacitTouch](https://pier-lab.github.io/tacittouch/) distills a task-trained
SimpleTouch teacher into the native vision-only π₀.₅ policy. The UniVTAC loader
reads the synchronized demonstrations. Both policies share the image
augmentation, instruction, state and normalized action representation. The
teacher also receives the two measured tactile RGB streams through its frozen
T3 encoder and tactile expert.

The training objective uses the student's Euler trajectory. Each batch
samples one of the ten query steps. The prefix runs without gradients. The
teacher and student use the same detached action state and flow time for
velocity MSE and raw post-block action-token MSE across 18 corresponding
action layers. An independent student flow-matching pass uses demonstrated
actions with freshly sampled noise and time. The default weights for velocity,
features and demonstrated actions are 1, 0.01 and 1, respectively.

## Train

Use the [SimpleTouch policy environment](../../README.md)
and the [UniVTAC demonstrations and assets](../../README.md#univtac-example).
Place each task's episodes under `<data-root>/<task>/clean/`. Use the task's
normalization statistics and tokenizer from the teacher's training contract.
For a released SimpleTouch task, its teacher can be fetched with
`python scripts/simpletouch/download_checkpoint.py --task "$TACITTOUCH_TASK"`.

From the repository root with the policy environment activated:

```bash
export TACITTOUCH_TASK=lift_bottle
export SIMPLETOUCH_DATA_ROOT=/path/to/univtac/dataset
export SIMPLETOUCH_NORM_STATS_PATH=/path/to/lift_bottle/norm_stats.json
export SIMPLETOUCH_TOKENIZER_PATH=/path/to/paligemma_tokenizer.model
export SIMPLETOUCH_BASE_MODEL_PATH=/path/to/pi05_base_pytorch_rlinf
export SIMPLETOUCH_T3_PATH=/path/to/t3_large
export TACITTOUCH_TEACHER_CHECKPOINT=/path/to/lift_bottle/teacher/global_step_20000
bash scripts/tacittouch/train_univtac.sh
```

The checkpoint path must refer to a complete SimpleTouch teacher, including its
configuration, normalization assets and completion marker. The teacher is
loaded once per rank. TacitTouch trains both the visual-language backbone and
action expert and saves the student in the existing SimpleTouch checkpoint
layout. Training reads current tactile RGB directly from the demonstrations.

The main recipe uses episodes 0–49, eight FSDP ranks, a global batch of 64 and
20,000 AdamW updates. It uses a peak learning rate of `5e-5`, 1,000 warmup
steps and a cosine decay to `5e-6`. Lift Bottle and Put Bottle use head RGB.
Lift Can and Insert Tube use head and wrist RGB. All models use the task
instruction, proprioception and a 50-action prediction horizon.

The launcher also accepts the remaining UniVTAC tasks when their
demonstrations, statistics and task-trained teacher are available. For
loss ablations, set `actor.model.tacittouch.feature_loss_weight=0` or
`actor.model.tacittouch.demonstration_loss_weight=0` as a Hydra override.

## Evaluate

Set up the [UniVTAC environment](../../README.md#-evaluation) and download a
student checkpoint from the [shared model repository](https://huggingface.co/cyang203912/SimpleTouch/tree/main/tacittouch)
using the [download example](../../README.md#distillation). For a checkpoint
downloaded to `./checkpoints`, run the following commands.

```bash
export TACITTOUCH_TASK=lift_bottle
export TACITTOUCH_CHECKPOINT="$PWD/checkpoints/tacittouch/${TACITTOUCH_TASK}/global_step_20000"
bash scripts/tacittouch/evaluate_univtac.sh
```

The evaluator uses the student with camera images, the instruction and
proprioception, and ten Euler steps per predicted chunk. The default execution
horizons are 25 actions for Insert Tube and Insert Hole, and 50 for the other
tasks. The TacitTouch reference evaluator applies the 5 mm lateral tolerance
to the absolute displacements on both axes for Insert Tube. The student
checkpoint is sufficient for policy evaluation with the
standard tokenizer and normalization assets.
