"""Restricted action space for the INDOOR loop (relational composition).

The policy emits a JSON spec describing a room's contents relationally.
`render_indoor.py` writes this spec to disk and Infinigen's `rl_inject` hook
compiles it into the constraint-DSL (see
`src/infinigen_examples/constraints/rl_inject.py`).

There are TWO ways to describe an object, freely mixable in one `objects` list:

  * COUNT-BASED (anonymous set) -- the original, still supported::
        {"factory": "ChairFactory", "relation": "front_against",
         "target": "TableDiningFactory", "count": 4}

  * LABELED (a named singleton you can arrange individually)::
        {"id": "table",   "factory": "TableDiningFactory", "relation": "on_floor", "to": "room"}
        {"id": "chair_n", "factory": "ChairFactory", "relation": "front_against", "to": "table"}
    Each `id` becomes its own set, so two same-type objects can carry different
    constraints and be arranged relative to each other. `to` references another
    id, a factory name, or "room". A labeled object always has count 1.

Plus an optional top-level `arrange` list of soft geometric objectives between
labeled ids / factory names::
        {"type": "focus",    "a": "sofa",    "b": "tv",      "goal": "max", "weight": 2}
        {"type": "distance", "a": "chair_n", "b": "chair_s", "goal": "max", "weight": 3}

Kept deliberately small -- a curated whitelist of room types, factories,
relations and arrange types -- so the model's job is well defined and the reward
legible. Widen the whitelists (or move to raw-DSL) only after the loop runs end
to end. Every name is re-validated against Infinigen's registries by the hook;
the whitelists below are the *subset we advertise to the LLM*.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from typing import Optional

# ---- vocabulary (subset of the full Infinigen indoor vocabulary) ----

ROOM_TYPES: tuple[str, ...] = (
    "DiningRoom",
    "LivingRoom",
    "Bedroom",
    "Kitchen",
    "Bathroom",
)

# relations that attach an object to the ROOM (no target)
ROOM_RELATIONS: tuple[str, ...] = ("on_floor", "against_wall", "on_wall", "hanging")
# relations that attach an object to ANOTHER object (need a target / `to`)
OBJ_RELATIONS: tuple[str, ...] = ("ontop", "on", "front_against", "side_by_side")
ALL_RELATIONS: tuple[str, ...] = ROOM_RELATIONS + OBJ_RELATIONS

# soft geometric objectives; goal "min"/"max" picks minimize/maximize.
# (b is ignored for "center".)
ARRANGE_TYPES: tuple[str, ...] = (
    "distance", "spacing", "focus", "align", "symmetry", "center",
)

# factory class names, grouped only for the schema doc; validation is flat.
FACTORIES: tuple[str, ...] = (
    # tables / seating
    "TableDiningFactory", "CoffeeTableFactory", "SideTableFactory", "SimpleDeskFactory",
    "ChairFactory", "OfficeChairFactory", "BarChairFactory",
    "SofaFactory", "ArmChairFactory", "BedFactory",
    # storage / appliances
    "SimpleBookcaseFactory", "LargeShelfFactory", "CellShelfFactory", "TVStandFactory",
    "KitchenIslandFactory", "OvenFactory", "MicrowaveFactory", "BeverageFridgeFactory",
    "ToiletFactory", "BathtubFactory", "StandingSinkFactory", "HardwareFactory",
    # lighting
    "CeilingLightFactory", "FloorLampFactory", "LampFactory", "DeskLampFactory",
    # decor / tabletop
    "WallArtFactory", "MirrorFactory", "RugFactory",
    "PlantContainerFactory", "VaseFactory",
    "BowlFactory", "PlateFactory", "WineglassFactory", "PotFactory",
)

_FACTORY_SET = set(FACTORIES)
_ROOM_SET = set(ROOM_TYPES)
_ROOM_REL_SET = set(ROOM_RELATIONS)
_OBJ_REL_SET = set(OBJ_RELATIONS)
_ALL_REL_SET = set(ALL_RELATIONS)
_ARRANGE_SET = set(ARRANGE_TYPES)

MAX_OBJECTS = 8  # fallback cap when a room's ontology budget is unavailable
MAX_ARRANGE = 6  # cap the soft-objective list


def _room_object_cap(room_type: str) -> int:
    """Per-room object budget from the ontology (bathroom=6, bedroom=8,
    living=10, ...), falling back to the global ``MAX_OBJECTS`` when the
    ontology can't be loaded. Keeps sanitize() consistent with the reward
    and author-check, which both budget per room."""
    try:
        from indoor_ontology import ONTOLOGY
        onto = ONTOLOGY.get(room_type)
        if onto is not None:
            return int(onto.max_objects)
    except Exception:
        pass
    return MAX_OBJECTS


@dataclass
class Placement:
    factory: str
    relation: str
    count: int = 1
    target: Optional[str] = None  # legacy alias for `to` when it is a factory
    id: Optional[str] = None      # label -> distinguishable singleton set
    to: Optional[str] = None      # "room" | another id | a factory name
    margin: Optional[float] = None  # accepted for forward-compat (not yet wired)


@dataclass
class Arrange:
    type: str
    a: str
    b: Optional[str] = None
    goal: Optional[str] = None      # "min" | "max" (hook picks a sensible default)
    weight: float = 1.0


@dataclass
class IndoorConfig:
    room_type: str
    seed: int
    objects: list[Placement] = field(default_factory=list)
    arrange: list[Arrange] = field(default_factory=list)

    def __post_init__(self):
        # Infinigen requires a uint32 seed; fold any int into range.
        self.seed = int(self.seed) % (2 ** 32)
        # tolerate objects/arrange arriving as plain dicts (JSON / asdict round-trip)
        self.objects = [
            o if isinstance(o, Placement) else Placement(**o) for o in self.objects
        ]
        self.arrange = [
            a if isinstance(a, Arrange) else Arrange(**a) for a in self.arrange
        ]

    # ---- (de)serialization ----

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "IndoorConfig":
        return cls(
            room_type=d["room_type"],
            seed=d.get("seed", 0),
            objects=[Placement(**o) for o in d.get("objects", [])],
            arrange=[Arrange(**a) for a in d.get("arrange", [])],
        )

    @classmethod
    def from_json(cls, s: str) -> "IndoorConfig":
        return cls.from_dict(json.loads(s))

    # ---- validation ----

    def sanitize(self) -> "IndoorConfig":
        """Drop invalid placements/arrangements; raise if room type is bad.

        Returns a cleaned copy so a partially-malformed model output still
        yields a runnable spec (mirrors the hook's tolerant behavior).
        """
        if self.room_type not in _ROOM_SET:
            raise ValueError(f"unknown room_type {self.room_type!r}")

        clean: list[Placement] = []
        ids: set[str] = set()
        cap = _room_object_cap(self.room_type)
        for o in self.objects[:cap]:
            if o.factory not in _FACTORY_SET:
                continue
            if o.relation not in _ALL_REL_SET:
                continue
            # normalize the parent reference: prefer `to`, fall back to `target`
            to = o.to or o.target
            if o.relation in _ROOM_REL_SET:
                to = "room"       # room relations never take a target
            elif to is None or (to not in _FACTORY_SET and to not in ids):
                # object relation needs a resolvable parent (a factory, or an
                # id defined by an EARLIER labeled placement)
                continue
            labeled = bool(o.id) and o.id not in ids
            if o.id and not labeled:
                continue          # duplicate id -> drop
            if labeled:
                ids.add(o.id)
                count = 1
            else:
                try:
                    count = max(1, min(8, int(o.count)))
                except (TypeError, ValueError):
                    count = 1
            clean.append(Placement(
                factory=o.factory, relation=o.relation, count=count,
                target=(to if to != "room" and to in _FACTORY_SET else None),
                id=(o.id if labeled else None),
                to=(None if to == "room" and not labeled else to),
                margin=o.margin,
            ))

        # arrange terms reference ids (defined above) or factory names
        resolvable = ids | _FACTORY_SET
        clean_arr: list[Arrange] = []
        for a in self.arrange[:MAX_ARRANGE]:
            if a.type not in _ARRANGE_SET:
                continue
            if a.a not in resolvable:
                continue
            needs_b = a.type != "center"
            if needs_b and a.b not in resolvable:
                continue
            goal = a.goal if a.goal in ("min", "max") else None
            try:
                weight = float(a.weight)
            except (TypeError, ValueError):
                weight = 1.0
            clean_arr.append(Arrange(
                type=a.type, a=a.a, b=(a.b if needs_b else None),
                goal=goal, weight=weight,
            ))

        return IndoorConfig(
            room_type=self.room_type, seed=self.seed,
            objects=clean, arrange=clean_arr,
        )

    @classmethod
    def random(cls, rng: random.Random | None = None) -> "IndoorConfig":
        r = rng or random
        rt = r.choice(ROOM_TYPES)
        # ~40% of the time emit a LABELED + arrange spec, otherwise count-based,
        # so the self-test / smoke exercises both schema branches.
        if r.random() < 0.4:
            objs = [
                Placement(id="anchor", factory=r.choice(FACTORIES),
                          relation="on_floor", to="room"),
                Placement(id="a", factory=r.choice(FACTORIES),
                          relation=r.choice(OBJ_RELATIONS), to="anchor"),
                Placement(id="b", factory=r.choice(FACTORIES),
                          relation=r.choice(OBJ_RELATIONS), to="anchor"),
            ]
            arr = [
                Arrange(type="focus", a="a", b="anchor", goal="max", weight=2.0),
                Arrange(type="distance", a="a", b="b", goal="max", weight=1.0),
            ]
            return cls(room_type=rt, seed=r.randint(0, 2**31 - 1),
                       objects=objs, arrange=arr)

        n = r.randint(2, 5)
        objs: list[Placement] = []
        placed_floor: list[str] = []
        for _ in range(n):
            if placed_floor and r.random() < 0.4:
                objs.append(Placement(
                    factory=r.choice(FACTORIES),
                    relation=r.choice(OBJ_RELATIONS),
                    target=r.choice(placed_floor),
                    count=r.randint(1, 4),
                ))
            else:
                rel = r.choice(ROOM_RELATIONS)
                fac = r.choice(FACTORIES)
                if rel == "on_floor":
                    placed_floor.append(fac)
                objs.append(Placement(factory=fac, relation=rel, count=r.randint(1, 2)))
        return cls(room_type=rt, seed=r.randint(0, 2**31 - 1), objects=objs)


SCHEMA_FOR_LLM = f"""\
Return ONLY a JSON object describing the contents of one room, with keys:
- room_type: one of {list(ROOM_TYPES)}
- seed: integer in [0, 2147483647]
- objects: a list (max {MAX_OBJECTS}) of placements (see below)
- arrange: (optional) a list (max {MAX_ARRANGE}) of soft objectives (see below)

Each placement is one of two forms:
  COUNT-BASED (an anonymous group of identical objects):
    - factory: one of the classes below
    - relation: how it is placed
    - count: integer in [1, 8]
    - target: (ONLY for object-to-object relations) the factory it sits on/near
  LABELED (a single named object you can arrange individually):
    - id: a short unique label (e.g. "table", "chair_n")
    - factory: one of the classes below
    - relation: how it is placed
    - to: "room", OR the id of an EARLIER labeled object, OR a factory name
    (a labeled object is always a single instance)

Relations to the ROOM (to="room", or omit target): {list(ROOM_RELATIONS)}
    against_wall = STANDS ON THE FLOOR with its back to a wall (tubs, toilets, sinks, shelves, plants)
    on_floor = stands on the floor away from walls (freestanding)
    on_wall = MOUNTED ON A WALL at eye height (mirrors, wall art, towel hardware)
    hanging = hangs from the CEILING (ceiling lights only)
Relations to ANOTHER object (needs target / to): {list(OBJ_RELATIONS)}
    ontop = sits on its top; on = on its support surface; front_against = faces it; side_by_side

Each arrange objective (all reference an id or a factory name):
    - type: one of {list(ARRANGE_TYPES)}
        distance = push a and b apart; spacing = keep 2D clearance; focus = orient a toward b;
        align = align a and b's facing; symmetry = make a symmetric about b; center = center a on its surface
    - a: id or factory name
    - b: id or factory name (not needed for "center")
    - goal: "min" or "max" (optional; a sensible default is chosen)
    - weight: number (optional, default 1)

Factories: {list(FACTORIES)}

Example (labeled + arrange):
{{"room_type": "DiningRoom", "seed": 42,
  "objects": [
    {{"id": "table", "factory": "TableDiningFactory", "relation": "on_floor", "to": "room"}},
    {{"factory": "ChairFactory", "relation": "front_against", "target": "TableDiningFactory", "count": 4}},
    {{"id": "tv", "factory": "TVStandFactory", "relation": "against_wall", "to": "room"}},
    {{"id": "art", "factory": "WallArtFactory", "relation": "on_wall", "to": "room"}},
    {{"factory": "CeilingLightFactory", "relation": "hanging", "count": 1}}
  ],
  "arrange": [
    {{"type": "focus", "a": "table", "b": "tv", "goal": "max", "weight": 2}}
  ]}}
"""


if __name__ == "__main__":
    # quick self-test: random specs (both branches) round-trip + sanitize clean
    for s in range(6):
        cfg = IndoorConfig.random(random.Random(s))
        back = IndoorConfig.from_json(cfg.to_json()).sanitize()
        assert back.room_type in _ROOM_SET
        print(f"seed={s} objects={len(back.objects)} arrange={len(back.arrange)}")
    cfg = IndoorConfig.random(random.Random(0))
    print("--- example ---")
    print(cfg.to_json())
    print("--- sanitized ---")
    print(IndoorConfig.from_json(cfg.to_json()).sanitize().to_json())
