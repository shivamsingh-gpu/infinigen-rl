#!/usr/bin/env bash
#SBATCH --job-name=indoor-sample-render
#SBATCH --partition=p4-80-main
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=03:00:00
#SBATCH --output=/nara-efs/marketing/shhsing/infinigen/rl_infinigen_beginner/sample_renders/slurm-%j.out
set -xuo pipefail
cd /nara-efs/marketing/shhsing/infinigen/rl_infinigen_beginner
mkdir -p sample_renders
source venv/bin/activate
cd scripts
export RL_INDOOR_CAMERA_CORNER=1
export RL_INDOOR_EXCLUSIVE=1
export RL_INDOOR_SAMPLES=${RL_INDOOR_SAMPLES:-200}
export RL_INDOOR_BATH_HINGE=${RL_INDOOR_BATH_HINGE:-0.35}
export RL_INDOOR_BATH_NARROW=${RL_INDOOR_BATH_NARROW:-1.7}
python render_samples.py --samples sample_step60.json --workers "${RL_RENDER_WORKERS:-8}"
echo "SCRIPT_EXIT=$?"
