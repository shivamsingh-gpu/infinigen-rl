"""verl custom reward for the Infinigen indoor RL loop (schema-vs-ground-truth),
room-type generalized.

Wired into verl via::

    reward.custom_reward_function.path=<abs path to this file>
    reward.custom_reward_function.name=compute_score

The naive reward manager calls
``compute_score(data_source=, solution_str=, ground_truth=, extra_info=)`` per
sample (see verl/workers/reward_manager/naive.py). We return a **dict** whose
``"score"`` is the scalar reward and whose other keys are logged as extra
metrics (they stream to TensorBoard), so every sub-reward is visible.

This is the room-agnostic successor to ``reward_bathroom.py``: identical scoring
math, but the ontology (which factories are core/optional/forbidden, the
relation classes, and the object budget ``MAX_OBJECTS``) is looked up per sample
from ``indoor_ontology.ONTOLOGY[room_type]``, where ``room_type`` comes from the
GROUND TRUTH schema (falling back to the policy's own emitted room_type, then to
a neutral fallback ontology). One reward file therefore grades bathroom, bedroom
and any future room type; the parquet can be single-type or merged.

The reward is a pure comparison of the policy's emitted `IndoorConfig` JSON
against the ground-truth schema for the same reference -- NO rendering, no VLM,
fully deterministic. Design (all terms in [0,1] before weighting):

    reward = clip( w_P*presence            # GT's core fixtures reproduced
                 + w_L*placement           # objects on the right surface class
                 + w_S*set_match           # factory set matches GT (F1)
                 + optional_bonus          # good extras that GT also has
                 - forbidden_penalty       # objects that don't belong
                 - shelf_penalty           # food/tableware inside storage
                 - clutter_penalty         # more than MAX_OBJECTS objects
                 , 0.0, 1.0)

Unparseable / schema-invalid output earns 0.0 (parse_ok=0).
"""
from __future__ import annotations

import json
import os
import sys

# Make sibling modules importable regardless of verl's import machinery
# (verl loads this file by absolute path, so its dir may not be on sys.path).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from indoor_config_space import IndoorConfig  # noqa: E402
import indoor_ontology as onto_reg  # noqa: E402

# ---- reward weights (positive terms sum to 1.0 at their maxima) ----
W_PRESENCE = 0.45
W_PLACEMENT = 0.25
W_SET_MATCH = 0.20
MAX_OPTIONAL_BONUS = 0.10
OPTIONAL_BONUS_PER = 0.05
FORBIDDEN_PER = 0.30      # per forbidden object instance
SHELF_PER = 0.30         # per food/tableware item sitting in storage
CLUTTER_PER = 0.05       # per object past the room's MAX_OBJECTS budget


# --------------------------------------------------------------------------- #
# JSON extraction                                                             #
# --------------------------------------------------------------------------- #
def _extract_json(text: str) -> dict | None:
    """Pull the schema object out of a model response.

    Tolerates leading reasoning, ```json fences, and trailing prose by scanning
    for the LAST top-level ``{...}`` span that parses as JSON.
    """
    if not text:
        return None
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass
    starts: list[int] = []
    best: dict | None = None
    for i, ch in enumerate(text):
        if ch == "{":
            starts.append(i)
        elif ch == "}" and starts:
            start = starts.pop()
            if not starts:  # a top-level object closed
                candidate = text[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict) and "room_type" in obj:
                        best = obj  # keep last valid one
                except (json.JSONDecodeError, ValueError):
                    continue
    return best


def _zero(reason: str) -> dict:
    return {
        "score": 0.0,
        "parse_ok": 0.0,
        "presence": 0.0,
        "placement": 0.0,
        "set_match": 0.0,
        "forbidden_penalty": 0.0,
        "shelf_penalty": 0.0,
        "clutter_penalty": 0.0,
        "n_forbidden": 0.0,
        "n_objects": 0.0,
    }


# --------------------------------------------------------------------------- #
# sub-scores (all take the resolved RoomOntology `onto`)                      #
# --------------------------------------------------------------------------- #
def _id_to_factory(cfg: IndoorConfig) -> dict[str, str]:
    return {o.id: o.factory for o in cfg.objects if o.id}


def _factory_set(cfg: IndoorConfig) -> set[str]:
    return {o.factory for o in cfg.objects}


def _presence(gen: IndoorConfig, gt: IndoorConfig | None, onto) -> float:
    """Fraction of the GROUND TRUTH's core fixtures the policy reproduced.

    The reference decides which core fixtures belong. Without a GT we can only
    demand the always-expected fixtures. Extra core fixtures the GT does not
    have are NOT rewarded here -- they show up as set_match precision loss.
    """
    present = _factory_set(gen)
    if gt is not None:
        # The reference decides which core fixtures belong. If the GT has none
        # (e.g. an armchair-cluster living room with no sofa), the reference
        # does not demand one -> presence is not penalized. Do NOT fall back to
        # `always` here: a sofa-less GT would otherwise score 0 against itself.
        want = _factory_set(gt) & onto.core_fixtures
    else:
        want = set(onto.always)  # no reference -> demand the always-expected set
    if not want:
        return 1.0  # no fixture expectation for this room -> presence not penalized
    return sum(1 for f in want if f in present) / len(want)


def _optional_bonus(gen: IndoorConfig, gt: IndoorConfig, onto) -> float:
    gen_opt = _factory_set(gen) & onto.optional_good
    gt_opt = _factory_set(gt) & onto.optional_good
    shared = gen_opt & gt_opt
    return min(MAX_OPTIONAL_BONUS, OPTIONAL_BONUS_PER * len(shared))


def _placement(gen: IndoorConfig, onto) -> float:
    """Mean placement credit over object instances.

    1.0  relation is wall/ceiling-aligned AND the right class for the factory,
         object-anchored (ontop/on/front_against/side_by_side), OR ``on_floor``
         for a floor_any factory (freestanding is correct for a sofa/table/rug).
    0.5  ``on_floor`` for a factory the room has no strong opinion about.
    0.0  wrong surface class (mirror hanging from the ceiling, bed on_wall, ...).
    """
    total = 0.0
    credit = 0.0
    for o in gen.objects:
        n = max(1, int(o.count or 1))
        total += n
        ok = onto.relation_ok(o.factory, o.relation)
        if not ok:
            continue
        if o.relation in onto_reg.WALL_ALIGNED_RELATIONS:
            credit += n
        elif o.relation == "on_floor":
            # Freestanding is the CORRECT placement for floor_any furniture
            # (sofas, coffee/side tables, rugs, plants, chairs) -> full credit.
            # floor_wall_only items on_floor never reach here (relation_ok rejects
            # it); the 0.5 covers factories the room has no strong opinion about.
            credit += n if o.factory in onto.floor_any else 0.5 * n
        else:  # anchored to another object
            credit += n
    return (credit / total) if total else 0.0


def _clutter(gen: IndoorConfig, onto) -> tuple[float, int]:
    n = sum(max(1, int(o.count or 1)) for o in gen.objects)
    return min(1.0, CLUTTER_PER * max(0, n - onto.max_objects)), n


def _set_match_f1(gen: IndoorConfig, gt: IndoorConfig) -> float:
    g, t = _factory_set(gen), _factory_set(gt)
    if not g and not t:
        return 1.0
    if not g or not t:
        return 0.0
    inter = len(g & t)
    if inter == 0:
        return 0.0
    precision = inter / len(g)
    recall = inter / len(t)
    return 2 * precision * recall / (precision + recall)


def _forbidden(gen: IndoorConfig, onto) -> tuple[float, int]:
    n = sum(
        max(1, int(o.count or 1))
        for o in gen.objects
        if o.factory in onto.forbidden
    )
    return min(1.0, FORBIDDEN_PER * n), n


def _shelf_content(gen: IndoorConfig) -> tuple[float, int]:
    """Penalize food/tableware placed on/inside storage furniture."""
    id2fac = _id_to_factory(gen)
    n = 0
    for o in gen.objects:
        if o.factory not in onto_reg.FOOD_TABLEWARE:
            continue
        if o.relation not in onto_reg.ON_RELATIONS:
            continue
        parent = o.to or o.target
        if parent is None:
            continue
        parent_fac = id2fac.get(parent, parent)  # id -> its factory, else literal
        if parent_fac in onto_reg.STORAGE:
            n += max(1, int(o.count or 1))
    return min(1.0, SHELF_PER * n), n


# --------------------------------------------------------------------------- #
# entry point                                                                 #
# --------------------------------------------------------------------------- #
def compute_score(data_source=None, solution_str=None, ground_truth=None,
                  extra_info=None, **kwargs) -> dict:
    obj = _extract_json(solution_str or "")
    if obj is None:
        return _zero("no-json")
    try:
        gen = IndoorConfig.from_dict(obj).sanitize()
    except (KeyError, TypeError, ValueError):
        return _zero("invalid-schema")

    # ground_truth is the GT schema JSON string (or dict); parse tolerantly.
    gt_cfg = None
    if ground_truth is not None:
        try:
            gt_obj = ground_truth if isinstance(ground_truth, dict) else json.loads(ground_truth)
            gt_cfg = IndoorConfig.from_dict(gt_obj).sanitize()
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            gt_cfg = None

    # Ontology dispatches on the GT's room type (the reference decides the room);
    # fall back to the policy's own room_type, then to a neutral ontology.
    room_type = (gt_cfg.room_type if gt_cfg is not None else None) or getattr(gen, "room_type", None)
    onto = onto_reg.get_or_fallback(room_type)

    presence = _presence(gen, gt_cfg, onto)
    placement = _placement(gen, onto)
    clutter_pen, n_total = _clutter(gen, onto)
    set_match = _set_match_f1(gen, gt_cfg) if gt_cfg is not None else 0.0
    bonus = _optional_bonus(gen, gt_cfg, onto) if gt_cfg is not None else 0.0
    forbidden_pen, n_forbidden = _forbidden(gen, onto)
    shelf_pen, _n_shelf = _shelf_content(gen)

    raw = (
        W_PRESENCE * presence
        + W_PLACEMENT * placement
        + W_SET_MATCH * set_match
        + bonus
        - forbidden_pen
        - shelf_pen
        - clutter_pen
    )
    score = max(0.0, min(1.0, raw))

    return {
        "score": score,
        "parse_ok": 1.0,
        "presence": presence,
        "placement": placement,
        "set_match": set_match,
        "optional_bonus": bonus,
        "forbidden_penalty": forbidden_pen,
        "shelf_penalty": shelf_pen,
        "clutter_penalty": clutter_pen,
        "n_forbidden": float(n_forbidden),
        "n_objects": float(n_total),
    }


if __name__ == "__main__":
    # smoke test across room types: GT-vs-itself ~1, junk 0, forbidden drops.
    bath_gt = json.dumps({
        "room_type": "Bathroom", "seed": 1,
        "objects": [
            {"id": "tub", "factory": "BathtubFactory", "relation": "against_wall", "to": "room"},
            {"id": "toilet", "factory": "ToiletFactory", "relation": "against_wall", "to": "room"},
            {"id": "sink", "factory": "StandingSinkFactory", "relation": "against_wall", "to": "room"},
            {"id": "mirror", "factory": "MirrorFactory", "relation": "on_wall", "to": "room"},
        ],
    })
    print("bathroom GT vs GT:", compute_score(solution_str=bath_gt, ground_truth=bath_gt)["score"])

    bed_gt = json.dumps({
        "room_type": "Bedroom", "seed": 1,
        "objects": [
            {"id": "bed", "factory": "BedFactory", "relation": "against_wall", "to": "room"},
            {"id": "night_l", "factory": "SideTableFactory", "relation": "against_wall", "to": "room"},
            {"id": "night_r", "factory": "SideTableFactory", "relation": "against_wall", "to": "room"},
            {"id": "wardrobe", "factory": "LargeShelfFactory", "relation": "against_wall", "to": "room"},
            {"id": "art", "factory": "WallArtFactory", "relation": "on_wall", "to": "room"},
            {"id": "rug", "factory": "RugFactory", "relation": "on_floor", "to": "room"},
            {"factory": "CeilingLightFactory", "relation": "hanging", "count": 1},
        ],
    })
    print("bedroom  GT vs GT:", compute_score(solution_str=bed_gt, ground_truth=bed_gt)["score"])

    # bathroom fixtures in a bedroom -> forbidden penalty
    bed_bad = json.dumps({
        "room_type": "Bedroom", "seed": 1,
        "objects": [
            {"id": "bed", "factory": "BedFactory", "relation": "against_wall", "to": "room"},
            {"id": "toilet", "factory": "ToiletFactory", "relation": "against_wall", "to": "room"},
        ],
    })
    print("bedroom w/ toilet vs bedroom GT:", compute_score(solution_str=bed_bad, ground_truth=bed_gt))
    print("bed as on_wall (bad placement):",
          compute_score(solution_str=bed_gt.replace('"against_wall"', '"on_wall"', 1),
                        ground_truth=bed_gt)["placement"])
    print("malformed:", compute_score(solution_str="not json", ground_truth=bed_gt)["score"])
