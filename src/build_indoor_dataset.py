"""Build the verl parquet dataset for the Infinigen indoor RL loop (any room
type, or several merged).

One row per reference: the policy is prompted (system rubric + user caption) to
emit an `IndoorConfig` JSON; the row carries the ground-truth schema in
`reward_model.ground_truth` so `reward_indoor.compute_score` can grade it. The
reward dispatches on the GT's room_type, so a single reward file grades a
single-room or a merged parquet.

Row schema matches verl's convention (cf. examples/data_preprocess/gsm8k.py):
    data_source, prompt=[{role,content}...], ability,
    reward_model={style, ground_truth}, extra_info={...}

Run:  python build_indoor_dataset.py --room bedroom \
          --out_dir /nara-efs/marketing/shhsing/verl/rl_infinigen_bedroom/data
      python build_indoor_dataset.py --rooms bathroom bedroom --out_dir <merged data dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from indoor_config_space import IndoorConfig, SCHEMA_FOR_LLM
import indoor_ontology as onto_reg

BASE = Path(__file__).resolve().parents[1]
DATA_SOURCE = "infinigen_indoor"

# --room value -> (Semantics room_type, references subdir suffix). Mirrors
# author_gt_schemas.ROOM_TYPE/REF_SUFFIX (bathroom refs live in ext_indoor_bath).
ROOM_TYPE = {"bathroom": "Bathroom", "bedroom": "Bedroom", "living": "LivingRoom",
             "kitchen": "Kitchen", "dining": "DiningRoom"}
REF_SUFFIX = {"bathroom": "bath", "bedroom": "bedroom", "living": "living",
              "kitchen": "kitchen", "dining": "dining"}


def system_prompt(room_type: str, onto: onto_reg.RoomOntology) -> str:
    """Room-parametrized layout rubric derived from the ontology."""
    room_lc = room_type.lower()
    always = sorted(onto.always)
    optional = sorted(onto.optional_good)
    # surface-class guidance from the relation-class sets
    floor_wall = sorted(onto.floor_wall_only)
    floor_any = sorted(onto.floor_any)
    on_wall = sorted(onto.wall_mount_only)
    ceiling = sorted(onto.ceiling_only)
    return f"""You are a 3D interior layout designer. Given a short description of a \
{room_lc}, output ONE JSON object describing its contents for the Infinigen scene generator.

{SCHEMA_FOR_LLM}

{room_type.upper()} RULES (obey strictly):
- room_type MUST be "{room_type}". Use at most {onto.max_objects} objects.
- Always include (unless the description clearly lacks it): {always}.
- Include ONLY items the description actually shows. You MAY add these when mentioned: {optional}.
- Surface class per object (to="room"):
  * "against_wall" (stands on the floor, back to a wall) for: {floor_wall}{
      ' and, optionally, ' + str(floor_any) if floor_any else ''}.
  * "on_floor" is allowed for freestanding items ({floor_any}) but prefer "against_wall".
  * "on_wall" (mounted on the wall) for: {on_wall}. Never put these on the floor or ceiling.
  * "hanging" (from the ceiling) for: {ceiling}. Never hang anything else.
- Give every object a short unique "id".
- NEVER include items that don't belong in a {room_lc}: {sorted(onto.forbidden)}. \
Do not put food or books on shelves.
- Output ONLY the JSON object, no prose.
"""


def build_row(room: str, room_type: str, gt_dir: Path, cap_dir: Path,
              stem: str, idx: int, split: str, sys_prompt: str) -> dict:
    schema = json.loads((gt_dir / f"{stem}.json").read_text())
    gt_json = IndoorConfig.from_dict(schema).sanitize().to_json()
    caption = (cap_dir / f"{stem}.txt").read_text().strip()
    user = f"Design the {room_type.lower()} described below.\n\n{caption}"
    return {
        "data_source": DATA_SOURCE,
        "prompt": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user},
        ],
        "ability": "layout",
        "reward_model": {"style": "rule", "ground_truth": gt_json},
        "extra_info": {"split": split, "index": idx, "ref_stem": stem,
                       "room_type": room_type},
    }


def collect_rooms(rooms: list[str], n_val: int) -> tuple[list[dict], list[dict], dict]:
    """Build train/val rows across one or more room types."""
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    val_by_room: dict[str, list[str]] = {}
    for room in rooms:
        room_type = ROOM_TYPE[room]
        onto = onto_reg.get(room_type)
        if onto is None:
            raise SystemExit(f"no ontology for {room_type!r}")
        gt_dir = BASE / "gt_schemas" / room
        cap_dir = gt_dir / "captions"
        ref_dir = BASE / "references" / f"ext_indoor_{REF_SUFFIX[room]}"
        sys_prompt = system_prompt(room_type, onto)

        stems = sorted(p.stem for p in ref_dir.glob("*.png")
                       if (gt_dir / f"{p.stem}.json").exists())
        if not stems:
            raise SystemExit(f"no GT schemas found for room {room!r} under {gt_dir}")
        # deterministic held-out split: every ~Nth reference -> val
        step = max(1, len(stems) // n_val)
        val_stems = set(stems[::step][:n_val])
        train_stems = [s for s in stems if s not in val_stems]
        val_by_room[room] = sorted(val_stems)

        for i, s in enumerate(train_stems):
            train_rows.append(build_row(room, room_type, gt_dir, cap_dir, s,
                                         len(train_rows), "train", sys_prompt))
        for s in sorted(val_stems):
            val_rows.append(build_row(room, room_type, gt_dir, cap_dir, s,
                                       len(val_rows), "val", sys_prompt))
    return train_rows, val_rows, val_by_room


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", choices=sorted(ROOM_TYPE),
                    help="single room type to build")
    ap.add_argument("--rooms", nargs="+", choices=sorted(ROOM_TYPE),
                    help="multiple room types merged into one parquet")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_val", type=int, default=5,
                    help="references held out for validation, per room type")
    args = ap.parse_args()

    rooms = args.rooms or ([args.room] if args.room else None)
    if not rooms:
        ap.error("pass --room <r> or --rooms <r1> <r2> ...")

    train_rows, val_rows, val_by_room = collect_rooms(rooms, args.n_val)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_rows).to_parquet(out / "train.parquet")
    pd.DataFrame(val_rows).to_parquet(out / "val.parquet")
    print(f"rooms={rooms}: wrote {len(train_rows)} train + {len(val_rows)} val rows -> {out}")
    for room, vs in val_by_room.items():
        print(f"  [{room}] val refs:", vs)


if __name__ == "__main__":
    main()
