#!/usr/bin/env bash
#SBATCH --job-name=indoor-sample
#SBATCH --partition=p4-80-main
#SBATCH --qos=batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=/nara-efs/marketing/shhsing/verl/rl_infinigen_bathroom/logs/sample-%j.out
set -xeuo pipefail
cd /nara-efs/marketing/shhsing/verl
COMPAT_DIR=/nara-efs/marketing/shhsing/verl/demo_qwen35_gsm8k/cuda_compat/usr/local/cuda-13.0/compat
[ -d "$COMPAT_DIR" ] && export LD_LIBRARY_PATH="${COMPAT_DIR}:${LD_LIBRARY_PATH:-}"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
.venv/bin/python /nara-efs/marketing/shhsing/infinigen/rl_infinigen_beginner/scripts/sample_indoor_policy.py \
  --model rl_infinigen_bathroom/ckpts/Qwen3.5-2B-GRPO-indoor/merged_hf_step60 \
  --parquet rl_infinigen_indoor/data/val.parquet \
  --n 15 --temperature 0.2 \
  --out rl_infinigen_bathroom/logs/sample_step60.json
