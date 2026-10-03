#!/usr/bin/env bash
# Durable GRPO training job (Qwen3.5-2B / Infinigen bathroom layout).
# Runs as its own batch allocation so it survives interactive-session teardown.
#SBATCH --job-name=qwen35-grpo-bathroom
#SBATCH --partition=p5-141-nara-preempt
#SBATCH --qos=team-nara-preempt
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/nara-efs/marketing/shhsing/verl/rl_infinigen_bathroom/logs/slurm-%j.out

set -xeuo pipefail
cd /nara-efs/marketing/shhsing/verl
bash rl_infinigen_bathroom/run_bathroom_grpo_fsdp.sh "$@"
echo "SCRIPT_EXIT=$?"
