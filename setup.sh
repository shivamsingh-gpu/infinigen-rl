#!/usr/bin/env bash
# One-time setup: verify the external deps and (optionally) apply the infinigen patches.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/config/paths.env"

echo "== checking vendored trainer =="
if [ -f "$VERL_SRC/verl/trainer/main_ppo.py" ]; then echo "  OK   vendored verl.trainer.main_ppo ($VERL_SRC)"; else
  echo "  MISS $VERL_SRC/verl/trainer/main_ppo.py  <-- vendored verl missing"; fi

echo "== checking external deps (weights/venv/renderer) =="
for d in "$INFINIGEN_ROOT" "$VENV" "$BASE_MODEL"; do
  [ -e "$d" ] && echo "  OK   $d" || echo "  MISS $d  <-- fix config/paths.env"
done

echo "== checking infinigen patch (rl_inject hook) =="
HOOK="$INFINIGEN_ROOT/src/infinigen_examples/constraints/rl_inject.py"
if [ -f "$HOOK" ]; then echo "  OK   rl_inject.py present"; else
  echo "  MISS rl_inject.py -- copy patches/infinigen/rl_inject.py there and apply the"
  echo "       home.py / generate_indoors.py edits from docs/INFINIGEN_PATCHES.md"
fi
grep -q "maybe_inject_rl_constraints" "$INFINIGEN_ROOT/src/infinigen_examples/constraints/home.py" 2>/dev/null \
  && echo "  OK   home.py hook present" || echo "  MISS home.py hook (see docs/INFINIGEN_PATCHES.md)"

echo "== reward import smoke test =="
PYTHONPATH="$HERE/src" "$PY" -c "import indoor_config_space, indoor_ontology, reward_indoor; print('  OK   reward modules import')" \
  || echo "  FAIL reward import -- check \$VENV"
echo "done. next: see docs/TRAINING.md"
