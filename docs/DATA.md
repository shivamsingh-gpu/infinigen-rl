# Data

## Sources (symlinked into the repo)
- `data_gt_schemas/<room>/*.json` — the GT `IndoorConfig` schemas (191 total across 5 rooms),
  authored by `src/author_gt_schemas.py`. Captions live alongside under `captions/`.
- `data_references/ext_indoor_<suffix>/*.png` — the reference photos (suffix: `bath`, `bedroom`,
  `living`, `kitchen`, `dining`). Used for GT authoring and eval contact sheets, **not** training.
- `data_parquet/{train,val}.parquet` — the built dataset the trainer reads.

## Counts (merged policy)
| room | train | val |
|------|-------|-----|
| bathroom | 38 | 5 |
| bedroom | 45 | 5 |
| living | 54 | 5 |
| kitchen | 18 | 5 |
| dining | 11 | 5 |
| **merged** | **166** | **25** |

## Parquet row schema
```
data_source  = "infinigen_indoor"
prompt       = [ {role:"system", content:<SCHEMA_FOR_LLM spec>},
                 {role:"user",   content:"Design the <room> described below.\n\n<caption>"} ]
ability      = "layout"
reward_model = { style:"rule", ground_truth:<GT IndoorConfig JSON string> }
extra_info   = { split, index, ref_stem, room_type }
```
The policy conditions on `prompt`; the reward reads `reward_model.ground_truth`; `extra_info` is
bookkeeping. Nothing else is passed to the model.

## Build / rebuild
```bash
source config/paths.env
$PY src/build_indoor_dataset.py --rooms $ROOMS --out_dir $DATA_ROOT --n_val 5
# single room:   --room bathroom
```
`--n_val` (default 5) rows per room are held out for validation. `data.shuffle=True` at train time,
so the 166 train rows are reshuffled each epoch.

## Adding a room type
1. Extend `indoor_ontology.ONTOLOGY` with the new room's core/optional/forbidden factories, surfaces, budget.
2. Author GT schemas with `author_gt_schemas.py --room <new>` (+ reference photos under `references/ext_indoor_<new>`).
3. Add per-room sizing/camera to `render_indoor.ROOM_SIZING` if rendering.
4. Rebuild the parquet with the new room in `--rooms`.
