"""Validate + normalize the hand-authored indoor ground-truth schemas.

Room-type generalized (default: bathroom). Each reference in
references/ext_indoor_<room>/<stem>.png must have a matching
gt_schemas/<room>/<stem>.json authored by Claude. This tool:

  * loads every schema through IndoorConfig.from_dict().sanitize() -- any
    out-of-vocabulary factory / relation is dropped by sanitize(), so we diff
    the raw vs sanitized object list and FLAG drops (they mean the author used
    a name outside the action space);
  * enforces the room rubric from indoor_ontology.ONTOLOGY[<RoomType>] (fixtures
    follow the reference): the room's always-expected fixtures are present, no
    FORBIDDEN factories, every object uses the right surface class for its
    factory (floor fixtures against_wall, mirrors/art/hardware on_wall, ceiling
    lights hanging -- see RoomOntology.relation_ok), at most MAX_OBJECTS objects,
    no food/tableware on storage; WARNS when a fixture is in the schema but the
    caption never mentions it (per-room _CAPTION_HINTS);
  * rewrites the file with the normalized (sanitized) JSON so the stored GT is
    always exactly what the reward will parse.

Run:  python author_gt_schemas.py --room bedroom          # validate + normalize
      python author_gt_schemas.py --room bedroom --check  # validate only, nonzero exit on error
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from indoor_config_space import IndoorConfig
import indoor_ontology as onto_reg

_BASE = Path(__file__).resolve().parents[1]

# --room value -> (Semantics room_type, references subdir suffix). The bathroom
# reference dir is historically ``ext_indoor_bath`` (not ``ext_indoor_bathroom``).
ROOM_TYPE = {
    "bathroom": "Bathroom",
    "bedroom": "Bedroom",
    "living": "LivingRoom",
    "kitchen": "Kitchen",
    "dining": "DiningRoom",
}
REF_SUFFIX = {
    "bathroom": "bath",
    "bedroom": "bedroom",
    "living": "living",
    "kitchen": "kitchen",
    "dining": "dining",
}

# Fixtures whose presence in the schema should be corroborated by the caption.
_CAPTION_HINTS = {
    "Bathroom": {
        "BathtubFactory": ("tub", "bath ", "bathing", "soaking"),
        "ToiletFactory": ("toilet", "wc", "water closet"),
    },
    "Bedroom": {
        "BedFactory": ("bed", "sleep", "mattress"),
    },
    "LivingRoom": {
        "SofaFactory": ("sofa", "couch", "sectional", "settee", "loveseat"),
    },
    "Kitchen": {
        "OvenFactory": ("oven", "stove", "range", "cooktop", "cooker"),
        "BeverageFridgeFactory": ("fridge", "refrigerator"),
        "KitchenIslandFactory": ("island",),
    },
    "DiningRoom": {
        "TableDiningFactory": ("dining table", "table", "dining"),
    },
}


def check_one(stem: str, raw: dict, onto, cap_dir: Path,
              caption_hints: dict) -> tuple[IndoorConfig, list[str], list[str]]:
    """Return (sanitized cfg, errors, warnings) for one schema dict."""
    errors: list[str] = []
    warnings: list[str] = []

    n_raw = len(raw.get("objects", []))
    cfg = IndoorConfig.from_dict(raw).sanitize()
    if len(cfg.objects) != n_raw:
        warnings.append(
            f"{n_raw - len(cfg.objects)} placement(s) dropped by sanitize "
            f"(out-of-vocabulary factory/relation or unresolved parent)"
        )

    if cfg.room_type != onto.room_type:
        errors.append(f"room_type={cfg.room_type!r} (expected {onto.room_type!r})")

    facs = {o.factory for o in cfg.objects}
    missing = [f for f in onto.always if f not in facs]
    if missing:
        # a warning, not an error: the GT must stay faithful to the reference,
        # and some references genuinely lack an "always-expected" fixture.
        warnings.append(f"missing always-expected fixtures: {missing} -- confirm the reference lacks it")

    forbidden = sorted(facs & onto.forbidden)
    if forbidden:
        errors.append(f"forbidden factories present: {forbidden}")

    n_inst = sum(max(1, int(o.count or 1)) for o in cfg.objects)
    if n_inst > onto.max_objects:
        errors.append(f"{n_inst} objects > MAX_OBJECTS={onto.max_objects} (clutter)")

    # every object must use the surface class that fits its factory
    for o in cfg.objects:
        if not onto.relation_ok(o.factory, o.relation):
            errors.append(
                f"{o.factory} with relation {o.relation!r} "
                f"(allowed: {sorted(onto.allowed_room_relations(o.factory) or [])})"
            )
        elif o.relation == "on_floor":
            warnings.append(f"{o.factory} is on_floor (freestanding) -- intended?")

    # fixtures the caption never mentions -> probably invented
    cap_file = cap_dir / f"{stem}.txt"
    cap = cap_file.read_text().lower() if cap_file.exists() else ""
    for fac, hints in caption_hints.items():
        if fac in facs and cap and not any(h in cap for h in hints):
            warnings.append(f"{fac} in schema but caption never mentions {hints[0]!r}")

    # food/tableware on storage
    id2fac = {o.id: o.factory for o in cfg.objects if o.id}
    for o in cfg.objects:
        if o.factory in onto_reg.FOOD_TABLEWARE and o.relation in onto_reg.ON_RELATIONS:
            parent = id2fac.get(o.to or o.target, o.to or o.target)
            if parent in onto_reg.STORAGE:
                errors.append(f"{o.factory} placed in storage {parent}")

    # object-to-object PLACEMENT FEASIBILITY: an anchor the relation cannot
    # physically resolve against (e.g. `ontop` open shelving, or a chair
    # `front_against` a desk) makes the solver warn-and-continue, silently
    # dropping the object from the render. Catch it offline with a fix hint.
    for o in cfg.objects:
        tgt = o.to or o.target
        if not tgt or tgt == "room":
            continue
        anchor_fac = id2fac.get(tgt, tgt)  # id -> factory, else assume a factory name
        ok, suggestion, reason = onto_reg.relation_feasible(o.factory, o.relation, anchor_fac)
        if not ok:
            fix = f" -> use relation {suggestion!r}" if suggestion else ""
            errors.append(f"{o.factory} {o.relation!r} {anchor_fac}: {reason}{fix}")

    return cfg, errors, warnings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", default="bathroom", choices=sorted(ROOM_TYPE),
                    help="room type to validate (default: bathroom)")
    ap.add_argument("--check", action="store_true",
                    help="validate only; do not rewrite files")
    args = ap.parse_args()

    room_type = ROOM_TYPE[args.room]
    onto = onto_reg.get(room_type)
    if onto is None:
        print(f"no ontology for room_type {room_type!r}")
        return 1
    ref_dir = _BASE / "references" / f"ext_indoor_{REF_SUFFIX[args.room]}"
    gt_dir = _BASE / "gt_schemas" / args.room
    cap_dir = gt_dir / "captions"
    caption_hints = _CAPTION_HINTS.get(room_type, {})

    refs = sorted(p.stem for p in ref_dir.glob("*.png"))
    ok = 0
    n_err = 0
    missing_files = []
    for stem in refs:
        fp = gt_dir / f"{stem}.json"
        if not fp.exists():
            missing_files.append(stem)
            continue
        raw = json.loads(fp.read_text())
        cfg, errors, warnings = check_one(stem, raw, onto, cap_dir, caption_hints)
        if errors:
            n_err += 1
            print(f"[FAIL] {stem}")
            for e in errors:
                print(f"        ERROR: {e}")
        else:
            ok += 1
        for w in warnings:
            print(f"[warn] {stem}: {w}")
        if not args.check and not errors:
            fp.write_text(cfg.to_json() + "\n")

    print(f"\n[{args.room}] {ok}/{len(refs)} schemas valid; {n_err} failed; "
          f"{len(missing_files)} missing.")
    if missing_files:
        print("missing:", ", ".join(missing_files))
    return 1 if (n_err or missing_files) else 0


if __name__ == "__main__":
    sys.exit(main())
