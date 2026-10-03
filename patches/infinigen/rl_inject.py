# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""RL hook: compile a JSON scene spec into constraint-DSL and inject it.

Active only when the env var ``INFINIGEN_RL_SPEC`` points at a JSON file. When
unset (the normal case) this is a no-op, so stock Infinigen runs and the
``INFINIGEN_DEMO_BOX`` demo path are completely unaffected.

Two spec styles are supported (mix freely inside one ``objects`` list):

1. COUNT-BASED (anonymous sets)::

     {"factory": "ChairFactory", "relation": "front_against",
      "target": "TableDiningFactory", "count": 4}

   -> obj[ChairFactory].related_to(<table in room>, front_against).count().equals(4)

2. LABELED (named singleton instances you can arrange individually)::

     {"id": "table",   "factory": "TableDiningFactory", "relation": "on_floor", "to": "room"}
     {"id": "chair_n", "factory": "ChairFactory", "relation": "front_against", "to": "table"}

   Each ``id`` becomes its own factory subclass (a distinct FromGenerator tag),
   so it is a *distinguishable* singleton set. ``to`` references another id, a
   factory name, or "room". This is what lets the policy place two chairs of the
   same type under different constraints.

Plus an optional ``arrange`` list of soft geometric objectives between labeled
ids / factories::

     {"type": "focus",    "a": "sofa",   "b": "tv",     "goal": "max", "weight": 2}
     {"type": "distance", "a": "chair_n","b": "chair_s","goal": "max", "weight": 3}

ROOM RELATION SEMANTICS (what the schema words compile to):

* ``against_wall`` -> ``on_floor`` AND ``against_wall`` (stands on the floor with its
  back to a wall). This is the stock home.py idiom (``furniture.related_to(rooms,
  on_floor).related_to(rooms, against_wall)``). A bare wall relation leaves z as a
  free DOF and the object floats mid-wall -- that was the v1 "floating tub" bug.
* ``on_floor`` -> ``on_floor`` only (freestanding).
* ``on_wall`` -> ``flush_wall`` (wall-mounted: mirrors, wall art, towel hardware),
  plus an injected height band on the bottom edge (score + hard guard) because
  exclusive mode strips the stock ``wall_decorations`` height rules for objects
  it does not recognise.
* ``hanging`` -> top against the CEILING (ceiling lights only).
* ``flush_wall`` / ``spaced_wall`` / ``side_against_wall`` pass straight through.

Because tags in Infinigen attach to factory *classes* (via ``FromGenerator``),
per-instance identity is achieved by generating one alias subclass per id and
registering it in ``used_as`` before ``usage_lookup`` initializes -- see
``register_rl_aliases`` (called from ``home.py`` right after ``home_asset_usage``).
The alias inherits EVERY usage-set membership of its parent (Furniture,
WallDecoration, ...) so stock score terms that filter on Semantics still see it.

Unknown factories/relations/targets are dropped with a warning; the solver
warns-and-continues on anything unsatisfiable, so a partially-malformed spec
still yields a renderable room (low RL reward, not a crash).
"""

import json
import logging
import os

from infinigen.core.constraints import constraint_language as cl
from infinigen.core.tags import Semantics

from . import util as cu
from .semantics import home_asset_usage

logger = logging.getLogger(__name__)

# Relations attaching a child to the ROOM's surfaces (parent is the room; no target).
ROOM_RELATIONS = {
    "on_floor", "flush_wall", "against_wall", "spaced_wall",
    "side_against_wall", "hanging", "on_wall",
}
# Relations attaching a child to ANOTHER object (need a target/`to`).
OBJ_RELATIONS = {
    "ontop", "on", "front_against", "front_to_front",
    "side_by_side", "back_to_back", "leftright_leftright",
}
# Object-relations that are horizontal face-to-face contacts (check_z=False in
# constraints/util.py): they leave the child's height a free DOF, so the child
# must ALSO be `on_floor` to the room or it floats and never places. `ontop`/`on`
# are excluded -- they rest the child on a surface, which is its own z anchor.
# (front_against/front_to_front are intercepted earlier as seat-proximity, so
# they are not listed here; the remainder are same-height side/back contacts
# whose is_vertically_contained() check can pass.)
_FLOOR_OBJ_RELATIONS = {
    "side_by_side", "back_to_back", "leftright_leftright",
}

# Seat-to-furniture relations: a chair/stool pulled up to a table or island.
# The seat is TALLER than the (short) table/island anchor face, so the literal
# StableAgainst(front, side|front, check_z=False) can NEVER satisfy its
# is_vertically_contained() check -- the seat never places (confirmed via solver
# debug: `chairN failed relation StableAgainst(front,..) res=False`, a purely
# vertical non-containment). We compile these as `on_floor` + a soft distance
# pull toward the anchor instead (see _placement_factor / maybe_inject), which
# places the seats and clusters them around the table. The schema keeps its
# semantic relation word, so the policy target and schema-vs-schema reward are
# unchanged.
_SEAT_PROXIMITY_RELATIONS = {"front_against", "front_to_front"}
# weight of the per-seat distance-minimize pull toward its table/island anchor.
# 8 tucks all 4 chairs of a crowded dining table in (w=4 left one straggler
# ~1.5 m out); a 2-stool island is unaffected placement-wise (both place at any
# weight -- any visible-framing difference there is the deterministic corner
# camera, not the pull).
_SEAT_PROXIMITY_W = float(os.environ.get("RL_SEAT_PROXIMITY_W", "8"))

# arrange-type -> (constraint_language factory, needs_b, default goal).
# goal "min"/"max" selects .minimize()/.maximize(); a cost is minimized to
# satisfy it (align/symmetry), a score is maximized (focus/spacing).
_ARRANGE = {
    "distance": (cl.distance, True, "max"),
    # "spacing" (keep 2D clearance) is compiled to cl.distance, NOT cl.min_dist_2d:
    # min_dist_2d_impl passes `b` as a list of blender names but
    # trimesh_geometry.min_dist_2d() calls `b.to_planar()` directly (it only
    # resolves `a` from names) -> AttributeError on any object SET, crashing the
    # coarse solve. cl.distance -> min_dist() supports many-to-many name lists and
    # maximizing it pulls the two sets apart (same clearance effect). Render-only:
    # the schema keeps "spacing" and the schema-vs-schema reward is unchanged.
    "spacing": (cl.distance, True, "max"),
    "focus": (cl.focus_score, True, "max"),
    "align": (cl.angle_alignment_cost, True, "min"),
    "symmetry": (cl.reflectional_asymmetry, True, "min"),
    "center": (cl.center_stable_surface_dist, False, "min"),
}

# Module state shared between the two entry points within one process. Rebuilt
# on every register call so a stale spec never leaks across runs.
_SPEC = None
_ALIASES: dict = {}

# EXCLUSIVE mode (env INFINIGEN_RL_EXCLUSIVE or spec key "exclusive"): drop the
# stock per-room furniture rules so the render reflects ONLY the policy's
# placements -- the policy "owns" the room. This gives clean reward credit (the
# scene contains nothing the policy didn't ask for) at the cost of the policy
# having to compose the whole room itself. The room SHELL (walls/floor/ceiling/
# doors/windows) is unaffected -- it comes from the separate home_room_constraints
# problem. Lighting is kept regardless: a pitch-black render gives the reward
# nothing to see, and lighting is infrastructure, not furniture design.
#
# We also KEEP the stock *placement-quality* scores that add no objects but keep
# the ones we place sane: `furniture_aesthetics` (wall furniture spaced, fronts
# accessible) and `portal_accessibility` (doorways clear). They filter on
# Semantics tags, which our aliases inherit (see register_rl_aliases). The
# `furniture_fullness` (60-90 % floor coverage!) and every object-adding rule
# stay dropped.
# NB: `wall_decorations` is NOT kept -- its `wall_art.count().in_range(0, 6)` /
# `mirror.count().in_range(0, 1)` let the solver ADD stock art + a stock mirror
# (seen in the v2 smoke: a purple painting and a second mirror the schema never
# asked for). Height of our own on_wall objects is handled by rl_wall_height.
_EXCLUSIVE_KEEP_CONSTRAINTS = {"lighting", "ceiling_lights"}
_EXCLUSIVE_KEEP_SCORES = {"ceiling_lights", "furniture_aesthetics", "portal_accessibility"}

# Wall-mount height band for `on_wall` objects: bottom edge above the floor.
# Score band (hinge, weight 15) and a looser hard guard. Env-tunable.
_WALL_H_LO = float(os.environ.get("RL_INDOOR_WALL_H_LO", "0.9"))
_WALL_H_HI = float(os.environ.get("RL_INDOOR_WALL_H_HI", "1.6"))
_WALL_H_HARD_LO = float(os.environ.get("RL_INDOOR_WALL_H_HARD_LO", "0.6"))
_WALL_H_HARD_HI = float(os.environ.get("RL_INDOOR_WALL_H_HARD_HI", "1.9"))


def _load_spec():
    path = os.environ.get("INFINIGEN_RL_SPEC")
    if not path:
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"[rl] could not read INFINIGEN_RL_SPEC={path!r}: {e!r}")
        return None


def register_rl_aliases(used_as):
    """Generate one factory subclass per labeled `id` and register it in-place.

    Must run BEFORE ``usage_lookup.initialize_from_dict(used_as)`` so the alias
    factories are known to the solver. No-op when the RL env var is unset.

    The alias is added to EVERY usage set its parent belongs to (Object,
    RealPlaceholder, Furniture, WallDecoration, Storage, ...), so stock terms
    written against Semantics (e.g. `obj[Semantics.Furniture]`) apply to it.
    FromGenerator tags compare by exact class, so `obj[ParentFactory]` still does
    NOT match the alias -- that is the whole point of the alias.
    """
    global _SPEC, _ALIASES
    _SPEC = _load_spec()
    _ALIASES = {}
    if _SPEC is None:
        return

    fmap = {c.__name__: c for c in used_as[Semantics.Object]}

    for o in _SPEC.get("objects", []):
        oid = o.get("id")
        if not oid:
            continue  # count-based entries need no alias
        parent = fmap.get(o.get("factory"))
        if parent is None:
            logger.warning(f"[rl] id {oid!r}: unknown factory {o.get('factory')!r}; skipping")
            continue
        if oid in _ALIASES:
            logger.warning(f"[rl] duplicate id {oid!r}; skipping later definition")
            continue
        # a distinct subclass => distinct FromGenerator tag => distinguishable set
        alias = type(str(oid), (parent,), {})
        _ALIASES[oid] = alias
        n_sets = 0
        for sem, facs in used_as.items():
            if isinstance(facs, set) and parent in facs:
                facs.add(alias)
                n_sets += 1
        logger.debug(f"[rl] alias {oid!r} <- {parent.__name__} mirrored into {n_sets} usage sets")

    if _ALIASES:
        logger.info(f"[rl] registered {len(_ALIASES)} labeled alias factories: {list(_ALIASES)}")


def _resolve_set(name, obj, fmap):
    """id | factory-name -> the ObjectSetExpression for that set (unscoped)."""
    if name in _ALIASES:
        return obj[_ALIASES[name]]
    if name in fmap:
        return obj[fmap[name]]
    return None


def _room_chain(rel_name):
    """Room relation name -> fn(child_set, room) applying the compiled relation(s)."""
    if rel_name == "against_wall":
        return lambda c, r: c.related_to(r, cu.on_floor).related_to(r, cu.against_wall)
    if rel_name == "on_wall":
        return lambda c, r: c.related_to(r, cu.flush_wall)
    rel = getattr(cu, rel_name)
    return lambda c, r, rel=rel: c.related_to(r, rel)


def _placement_factor(o, obj, fmap):
    """Compile one `objects` entry into (count, fn(r)->domain, child_set, rel) or None."""
    oid = o.get("id")
    rel_name = o.get("relation")
    if rel_name not in ROOM_RELATIONS and rel_name not in OBJ_RELATIONS:
        logger.warning(f"[rl] unknown relation {rel_name!r}; dropping placement")
        return None

    if oid:
        child = obj[_ALIASES[oid]] if oid in _ALIASES else None
        count = 1
    else:
        child = _resolve_set(o.get("factory"), obj, fmap)
        count = int(o.get("count", 1))
    if child is None:
        logger.warning(f"[rl] could not resolve child for {o!r}; dropping")
        return None

    to = o.get("to") or o.get("target") or "room"

    if rel_name in ROOM_RELATIONS:
        if to != "room":
            logger.warning(f"[rl] room-relation {rel_name!r} ignores to={to!r}; using room")
        chain = _room_chain(rel_name)
        return count, (lambda r, c=child, ch=chain: ch(c, r)), child, rel_name

    # Seat pulled up to a table/island: compile as on_floor (always placeable);
    # a soft distance pull toward the anchor is added in maybe_inject so the
    # seats gather around it. The literal StableAgainst can't place a seat
    # taller than the anchor (see _SEAT_PROXIMITY_RELATIONS).
    if rel_name in _SEAT_PROXIMITY_RELATIONS and (to in _ALIASES or to in fmap):
        return count, (lambda r, c=child: c.related_to(r, cu.on_floor)), child, rel_name

    relation = getattr(cu, rel_name)
    # object relation -> parent is another set, scoped to the room.
    # Floor-standing contacts (side_by_side/back_to_back/leftright_leftright)
    # are horizontal face-to-face contacts with check_z=False, so they do NOT
    # anchor the child's height -- the child has a free z DOF and never places.
    # Stock pairs them with on_floor (home.py:536 scopes ALL furniture
    # `.related_to(rooms, on_floor)` before adding the contact relation), so we
    # must too. `ontop`/`on` rest the child on a surface (its own z anchor), so
    # they get NO on_floor.
    def _child_scoped(c, r):
        return c.related_to(r, cu.on_floor) if rel_name in _FLOOR_OBJ_RELATIONS else c

    if to in _ALIASES:
        return count, (lambda r, c=child, rel=relation, p=obj[_ALIASES[to]]:
                       _child_scoped(c, r).related_to(p.related_to(r), rel)), child, rel_name
    if to in fmap:
        return count, (lambda r, c=child, rel=relation, p=obj[fmap[to]]:
                       _child_scoped(c, r).related_to(p.related_to(r, cu.on_floor), rel)), child, rel_name
    logger.warning(f"[rl] object-relation {rel_name!r} needs a valid `to`, got {to!r}; dropping")
    return None


def _arrange_term(a, obj, fmap):
    """Compile one `arrange` entry into fn(r)->ScalarExpression, or None."""
    kind = a.get("type")
    spec = _ARRANGE.get(kind)
    if spec is None:
        logger.warning(f"[rl] unknown arrange type {kind!r}; dropping")
        return None
    fn, needs_b, default_goal = spec
    set_a = _resolve_set(a.get("a"), obj, fmap)
    if set_a is None:
        logger.warning(f"[rl] arrange {kind}: unknown a={a.get('a')!r}; dropping")
        return None
    set_b = None
    if needs_b:
        set_b = _resolve_set(a.get("b"), obj, fmap)
        if set_b is None:
            logger.warning(f"[rl] arrange {kind}: unknown b={a.get('b')!r}; dropping")
            return None
    goal = a.get("goal", default_goal)
    weight = float(a.get("weight", 1.0))

    def term(r):
        A = set_a.related_to(r)
        expr = fn(A, set_b.related_to(r)) if needs_b else fn(A)
        return expr.maximize(weight=weight) if goal == "max" else expr.minimize(weight=weight)

    return term


def _wall_height_terms(wall_sets):
    """(constraint_fn, score_fn) pinning `on_wall` sets' bottom edge into a height band."""
    def hard(r):
        expr = None
        for s in wall_sets:
            f = s.related_to(r, cu.flush_wall).all(
                lambda t: (t.distance(r, cu.floortags) > _WALL_H_HARD_LO)
                * (t.distance(r, cu.floortags) < _WALL_H_HARD_HI)
            )
            expr = f if expr is None else (expr * f)
        return expr

    def soft(r):
        expr = None
        for s in wall_sets:
            f = s.related_to(r, cu.flush_wall).mean(
                lambda t: t.distance(r, cu.floortags)
                .hinge(_WALL_H_LO, _WALL_H_HI)
                .minimize(weight=15)
            )
            expr = f if expr is None else (expr + f)
        return expr

    return hard, soft


def maybe_inject_rl_constraints(constraints, score_terms, rooms, obj):
    """If a spec is loaded, add its compiled hard constraints + arrange scores."""
    if _SPEC is None:
        return

    room_type = _SPEC.get("room_type", "DiningRoom")
    try:
        rt = getattr(Semantics, room_type)
    except AttributeError:
        logger.warning(f"[rl] unknown room_type {room_type!r}; skipping injection")
        return
    rt_rooms = rooms[rt]
    fmap = {c.__name__: c for c in home_asset_usage()[Semantics.Object]}

    factors = [f for f in (_placement_factor(o, obj, fmap)
                           for o in _SPEC.get("objects", [])) if f]

    # EXCLUSIVE mode: strip the stock furniture rules so only the policy's
    # placements populate the room -- the policy "owns" the room. We strip even
    # when the policy asked for nothing (factors == []): an empty spec then
    # renders a BARE room (walls/floor/ceiling only), which scores low and gives
    # the policy an honest gradient to start placing furniture. Placement-quality
    # scores in _EXCLUSIVE_KEEP_SCORES survive (they add no objects).
    exclusive = bool(os.environ.get("INFINIGEN_RL_EXCLUSIVE")) or bool(_SPEC.get("exclusive"))
    if exclusive:
        dropped_c = [k for k in list(constraints) if k not in _EXCLUSIVE_KEEP_CONSTRAINTS]
        dropped_s = [k for k in list(score_terms) if k not in _EXCLUSIVE_KEEP_SCORES]
        for k in dropped_c:
            del constraints[k]
        for k in dropped_s:
            del score_terms[k]
        logger.info(f"[rl] exclusive mode: dropped {len(dropped_c)} stock constraint(s) "
                    f"+ {len(dropped_s)} score term(s); kept constraints={list(constraints)} "
                    f"scores={list(score_terms)}")

    if exclusive and "CeilingLightFactory" in fmap:
        # stock `ceiling_lights` allows 1-4 per room and the solver happily
        # added 4 to a 7.5 m^2 bathroom (v2 smoke) -> cap at 1-2; the spec's own
        # labeled lights are a separate alias set and are not counted here.
        ceil = obj[fmap["CeilingLightFactory"]]
        spec_lights = any(
            o.get("factory") == "CeilingLightFactory" and o.get("relation") == "hanging"
            for o in _SPEC.get("objects", [])
        )
        lo, hi = (0, 1) if spec_lights else (1, 2)
        constraints["ceiling_lights"] = rt_rooms.all(
            lambda r, lo=lo, hi=hi: ceil.related_to(r, cu.hanging).count().in_range(lo, hi)
        )
        logger.info(f"[rl] exclusive mode: stock ceiling lights capped to [{lo},{hi}] per room")

    if factors:
        def per_room(r):
            expr = None
            for count, dom_fn, _child, _rel in factors:
                factor = dom_fn(r).count().equals(count)
                expr = factor if expr is None else (expr * factor)
            return expr
        constraints["rl_spec"] = rt_rooms.all(per_room)
        logger.info(f"[rl] injected {len(factors)} placement constraint(s) for {room_type}: "
                    + ", ".join(f"{o.get('id') or o.get('factory')}:{o.get('relation')}"
                                for o in _SPEC.get("objects", [])))

    # wall-mounted objects: keep their bottom edge in a sensible height band.
    wall_sets = [child for _c, _d, child, rel in factors if rel == "on_wall"]
    if wall_sets:
        hard, soft = _wall_height_terms(wall_sets)
        constraints["rl_wall_height"] = rt_rooms.all(hard)
        score_terms["rl_wall_height"] = rt_rooms.mean(soft)
        logger.info(f"[rl] injected wall-height band [{_WALL_H_LO},{_WALL_H_HI}] "
                    f"(hard [{_WALL_H_HARD_LO},{_WALL_H_HARD_HI}]) for {len(wall_sets)} on_wall set(s)")

    terms = [t for t in (_arrange_term(a, obj, fmap)
                         for a in _SPEC.get("arrange", [])) if t]
    for i, term in enumerate(terms):
        score_terms[f"rl_arrange_{i}"] = rt_rooms.mean(lambda r, tm=term: tm(r))
    if terms:
        logger.info(f"[rl] injected {len(terms)} arrange score term(s)")

    # Auto proximity for seat-to-furniture relations compiled as on_floor (see
    # _placement_factor): minimize each seat's distance to its table/island so
    # the floor-standing seats gather around the anchor instead of scattering;
    # collision keeps them from stacking, ringing them around it.
    seat_auto = []
    for o in _SPEC.get("objects", []):
        if o.get("relation") in _SEAT_PROXIMITY_RELATIONS:
            a = o.get("id") or o.get("factory")
            b = o.get("to") or o.get("target")
            if a and b and b != "room":
                seat_auto.append({"type": "distance", "a": a, "b": b,
                                  "goal": "min", "weight": _SEAT_PROXIMITY_W})
    seat_terms = [t for t in (_arrange_term(a, obj, fmap) for a in seat_auto) if t]
    for j, term in enumerate(seat_terms):
        score_terms[f"rl_seat_prox_{j}"] = rt_rooms.mean(lambda r, tm=term: tm(r))
    if seat_terms:
        logger.info(f"[rl] injected {len(seat_terms)} seat-proximity score term(s) "
                    f"(weight {_SEAT_PROXIMITY_W})")
