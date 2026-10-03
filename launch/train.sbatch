#!/usr/bin/env bash
# Durable GRPO training job (Qwen3.5-2B / Infinigen indoor layout).
# Runs as its own batch allocation so it survives interactive-session teardown.
# Uses the VENDORED verl trainer (external/verl) via config/paths.env.
#
#   sbatch launch/train.sbatch                       # defaults
#   sbatch ... launch/train.sbatch trainer.save_freq=20 trainer.test_freq=20
# NOTE: #SBATCH --output cannot expand env vars; if you relocate the repo, edit
#       the literal path below.
#SBATCH --job-name=qwen35-grpo-indoor
#SBATCH --partition=p4-80-main
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=08:00:00
#SBATCH --output=/nara-efs/marketing/shhsing/infinigen-rl/runs/logs/slurm-%j.out

set -xeuo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/../config/paths.env"
mkdir -p "$REPO_ROOT/runs/logs"
cd "$VERL_SRC"                       # cwd holds the vendored verl package
bash "$HERE/run_grpo_fsdp.sh" "$@"
echo "SCRIPT_EXIT=$?"
