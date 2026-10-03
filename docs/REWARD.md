# Reward

`src/reward_indoor.py` → `compute_score(data_source, solution_str, ground_truth, extra_info, **kw)`.

Parses the policy's generated JSON into an `IndoorConfig`, reads `room_type` from it, looks up
`indoor_ontology.ONTOLOGY[room_type]`, and scores the generated layout **against the GT schema**
(schema-vs-schema — no rendering). Returns a **dict**, so every sub-term streams to TensorBoard
as its own `val-aux/*` / reward curve.

## Formula (all terms in [0,1] before weighting)
```
reward = clip(  0.45 * presence          # GT's CORE fixtures reproduced
              + 0.25 * placement         # objects on the correct surface class
              + 0.20 * set_match         # F1 of generated factory set vs GT set
              + optional_bonus           # +0.05 each, capped 0.10, for sensible optional items
              - forbidden_penalty        # 0.30 per object that doesn't belong in the room
              - shelf_penalty            # 0.30 per food/tableware item stuffed into storage
              - clutter_penalty          # 0.05 per object past the room's budget
              , 0.0, 1.0)
```
Weights live at the top of `reward_indoor.py` (`W_PRESENCE`, `W_PLACEMENT`, `W_SET_MATCH`,
`FORBIDDEN_PER`, `SHELF_PER`, `CLUTTER_PER`, …).

## Sub-terms
| term | meaning | ideal |
|---|---|---|
| `parse_ok` | generated JSON parsed into a valid schema | 1.0 (0 ⇒ whole reward is 0) |
| `presence` | fraction of the room's CORE fixtures that appear | 1.0 |
| `placement` | mean credit for objects on the right surface class (floor/wall/surface) | 1.0 |
| `set_match` | F1 between generated factory set and GT factory set | → 1.0 |
| `optional_bonus` | small bonus for plausible optional items | up to 0.10 |
| `forbidden_penalty`, `n_forbidden` | objects that don't belong in this room | 0.0 |
| `shelf_penalty` | food/tableware placed inside storage furniture | 0.0 |
| `clutter_penalty`, `n_objects` | objects beyond the room budget / total object count | 0.0 / ~6 |

## Sanity check before any run (the reward ceiling)
GT scored against itself must be ≈1.0; junk must be 0.
```bash
source config/paths.env
$PY - <<'PY'
import glob, reward_indoor as R
for r in ['bathroom','bedroom','living','kitchen','dining']:
    xs=[]
    for p in glob.glob(f"$SCAFFOLD/gt_schemas/{r}/*.json".replace("$SCAFFOLD","$SCAFFOLD")):
        gt=open(p).read(); s=R.compute_score('infinigen_indoor',gt,gt,{})
        xs.append(s['score'] if isinstance(s,dict) else s)
    print(r,'min',round(min(xs),3),'mean',round(sum(xs)/len(xs),3),'n',len(xs))
print('junk', R.compute_score('infinigen_indoor','hello',gt,{})['score'])
PY
```
Last known: per-room GT-vs-GT min 0.95, means 0.99–1.0; junk → 0.0.

## How verl calls it
`run_grpo_fsdp.sh` sets `reward.custom_reward_function.path=<.../reward_indoor.py>` and
`...name=compute_score`. verl imports the file by path and calls it per sample. The returned
`score` is the scalar reward; the other dict keys are logged as metrics.

## Why schema-vs-schema (and the trade-off)
It is fast (no Blender in the 128-rollout loop) and deterministic (stable advantages). It rewards
*composition*, not *physical/visual realism* — that is checked by rendering in the eval phase.
See `ARCHITECTURE.md` → roadmap for render/VLM/physics reward extensions.
