#!/usr/bin/env bash
# Launch the two dashboards (localhost; tunnel in via ssh -L). Login nodes here
# are resource-starved so the fast loader panics -> force --load_fast=false.
set -euo pipefail
source "$(dirname "$0")/../config/paths.env"
EXP=${EXP:-Qwen3.5-2B-GRPO-indoor}
"$VENV/bin/tensorboard" --logdir "$TB_ROOT/$EXP" --host 127.0.0.1 --port 6007 --load_fast=false &
"$VENV/bin/tensorboard" --logdir "$VERL_ROOT/rl_infinigen_bathroom/tb_samples/$EXP" --host 127.0.0.1 --port 6008 --load_fast=false &
echo "training curves -> http://127.0.0.1:6007   sample gallery -> http://127.0.0.1:6008"
wait
