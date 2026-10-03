#!/usr/bin/env bash
# Durable GT render-validation job (Phase B): render an indoor room type's
# ground-truth schemas with Infinigen (eye-level corner camera, exclusive
# placement, predefined per-room footprint) and build reference-vs-render
# contact sheets + report.json. Runs as its own allocation so it survives
# interactive-session teardown.
#
#   sbatch render_gt.sbatch                                    # bathroom (default), all schemas
#   ROOM=bedroom sbatch render_gt.sbatch                       # bedroom, all schemas
#   ROOM=bedroom ONLY="12_x 33_y" sbatch render_gt.sbatch      # subset
#   RL_INDOOR_SAMPLES=512 sbatch render_gt.sbatch              # higher-quality pass
#
# Output goes to gt_validation/slurm-%j.out; the validator writes per-room
# artifacts under gt_validation/<room>/.
#
#SBATCH --job-name=indoor-gt-render-shh
#SBATCH --partition=p4-80-main
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=96
#SBATCH --time=06:00:00
#SBATCH --output=/nara-efs/marketing/shhsing/infinigen/rl_infinigen_beginner/gt_validation/slurm-%j.out

set -xuo pipefail
cd /nara-efs/marketing/shhsing/infinigen/rl_infinigen_beginner

# Driver venv (has PIL + the scaffold deps); the Infinigen subprocess itself
# uses infinigen/.venv, resolved inside render_scene.py (INFINIGEN_PY).
source venv/bin/activate
cd scripts

# Which room type to validate (bathroom | bedroom | ...); the validator reads ROOM.
export ROOM=${ROOM:-bathroom}

# Fixed corner camera (deterministic full-room view) is the render_indoor.py
# default; exclusive = only the schema's placements populate the room.
export RL_INDOOR_CAMERA_CORNER=1
export RL_INDOOR_EXCLUSIVE=1
export RL_INDOOR_SAMPLES=${RL_INDOOR_SAMPLES:-200}
# Room sizing comes from render_indoor.ROOM_SIZING (predefined floor plan);
# RL_INDOOR_ROOM_AREA overrides the derived area for any room if needed.
# Bathroom-only gin refinements (ignored for other room types):
export RL_INDOOR_BATH_HINGE=${RL_INDOOR_BATH_HINGE:-0.35}
export RL_INDOOR_BATH_NARROW=${RL_INDOOR_BATH_NARROW:-1.7}
# Doors are kept shut (populate_doors.all_closed) but the closed leaf is a big flat
# panel the corner/doorway eye catches broadside -> hide the door mesh so the render
# shows a clean wall opening instead of an intrusive dark slab. Override at submit.
export RL_INDOOR_CAMERA_HIDE_DOOR=${RL_INDOOR_CAMERA_HIDE_DOOR:-0}

ONLY_ARGS=()
if [[ -n "${ONLY:-}" ]]; then
  # shellcheck disable=SC2206
  ONLY_ARGS=(--only ${ONLY})
fi

python validate_gt_render.py --workers "${RL_RENDER_WORKERS:-8}" "${ONLY_ARGS[@]}"
echo "SCRIPT_EXIT=$?"
