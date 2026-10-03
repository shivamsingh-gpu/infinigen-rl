"""Per-room object ontology + placement rules shared by the reward and the
ground-truth authoring/validation tooling.

This is the room-type-generalized successor to ``bathroom_ontology.py`` (which
is now a thin shim re-exporting ``ONTOLOGY["Bathroom"]``). A ``RoomOntology``
bundles, for one room type: which factories are core fixtures, which are
optional-good, which are forbidden, which relation class each factory must use
(floor-standing vs wall-mounted vs ceiling-hung), and the object budget
(``MAX_OBJECTS``). The module-level ``STORAGE`` / ``FOOD_TABLEWARE`` sets and the
``WALL_ALIGNED_RELATIONS`` / ``ON_RELATIONS`` relation classes are shared across
all rooms.

Everything is expressed in terms of the factory-class names advertised in
``indoor_config_space.FACTORIES``, so if the action-space whitelist changes this
file is the one place to reconcile.

Add a new room type by appending a ``RoomOntology`` to ``ONTOLOGY``; the reward
(``reward_indoor.py``), the authoring validator (``author_gt_schemas.py``) and
the dataset builder (``build_indoor_dataset.py``) all dispatch on the GT's
``room_type`` and pick up the new entry automatically.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# shared (room-agnostic) sets                                                 #
# --------------------------------------------------------------------------- #
# Storage furniture: shelves / bookcases / closets. Placing food or tableware
# on these is explicitly penalized (the "shelves shouldn't have books or food"
# rule; there is no book factory in the whitelist, so food/tableware is what we
# can actually catch).
STORAGE: frozenset[str] = frozenset({
    "LargeShelfFactory", "CellShelfFactory", "SimpleBookcaseFactory",
})

# Food / tableware. Never appropriate as loose room clutter, and doubly bad on a
# storage shelf.
FOOD_TABLEWARE: frozenset[str] = frozenset({
    "BowlFactory", "PlateFactory", "WineglassFactory", "PotFactory",
})

# Relations that satisfy the "aligned to a wall/ceiling, not stray / not
# floating" rule for the placement sub-score.
WALL_ALIGNED_RELATIONS: frozenset[str] = frozenset({"against_wall", "on_wall", "hanging"})

# Object-to-object relations meaning "sitting on/inside this parent".
ON_RELATIONS: frozenset[str] = frozenset({"ontop", "on"})

# Object-to-object relations that are always fine as a ROOM-level relation check
# (they are judged by the parent object, not the room surface).
_OBJ_ANCHORED: frozenset[str] = frozenset({"front_against", "side_by_side"})

# --------------------------------------------------------------------------- #
# Object-to-object PLACEMENT FEASIBILITY (which anchor factories a relation can  #
# physically resolve against). These caused silent under-placement in the       #
# bedroom pilot -- the solver warns-and-continues, so a geometrically            #
# impossible relation just drops the object from the render. Encoded here so     #
# every room type inherits the guard (see relation_feasible + author --check).   #
#                                                                                #
# WHY these sets: Infinigen tags oriented-bbox faces (Top/Bottom/Front/Back/     #
# side) on every object, but the constraint-DSL relations key off DIFFERENT      #
# tags (see constraints/util.py):                                                #
#   ontop = StableAgainst(bottom, {Subpart.Top})   -> needs a flat TOP face      #
#   on    = StableAgainst(bottom, {Subpart.SupportSurface}) -> needs tag_support #
#   front_against  = StableAgainst(front, side)  -> child FRONT vs anchor SIDE    #
#                    (the barstool-at-island idiom, home.py:923)                 #
#   front_to_front = StableAgainst(front, front) -> the chair-at-desk idiom      #
#                    (home.py:731 uses this for OfficeChair+SimpleDesk)           #
# Open shelving (Cell/Large/Bookcase) tags SupportSurface on its cubby boards    #
# (`on`) but exposes no usable flat Top, so `ontop` a shelf silently fails;      #
# a chair `front_against` a desk fails because the desk's front (not side) is    #
# what faces the room -- it must be front_to_front.                              #

# Anchors that expose a usable flat top for `ontop` (proven to place in the
# bedroom renders + tag_support on their tops in util.py/home.py).
FLAT_TOP_ANCHORS: frozenset[str] = frozenset({
    "SideTableFactory", "CoffeeTableFactory", "SimpleDeskFactory",
    "TableDiningFactory", "TVStandFactory", "KitchenIslandFactory",
})
# Anchors that expose a SupportSurface for `on` (shelving cubby boards +
# counters/islands). Flat-tops also satisfy `on`, so the union is the `on` set.
SHELF_ON_ANCHORS: frozenset[str] = frozenset({
    "CellShelfFactory", "LargeShelfFactory", "SimpleBookcaseFactory",
    "KitchenIslandFactory",
})
# Seating that pulls up to a table/desk.
SEATING: frozenset[str] = frozenset({
    "OfficeChairFactory", "ChairFactory", "ArmChairFactory", "BarChairFactory",
})
# WALL-BACKED work surfaces a seat must face `front_to_front` (chair front faces
# the desk front, which points into the room). A desk sits against a wall so its
# SIDE faces sideways -- `front_against` (child-front-vs-anchor-side) leaves the
# seat unplaceable (stock home.py:731 uses front_to_front for OfficeChair+desk).
DESK_LIKE: frozenset[str] = frozenset({"SimpleDeskFactory"})
# Anchors whose SIDE a `front_against` child sits against: a barstool tucks against
# a kitchen island side, and dining chairs sit around a freestanding dining table's
# sides (stock home.py:1296-1301 uses front_against for diningchairs->TableDining).
FRONT_AGAINST_ANCHORS: frozenset[str] = frozenset({
    "KitchenIslandFactory", "TableDiningFactory",
})


def relation_feasible(child: str, relation: str, anchor: str) -> tuple[bool, str | None, str]:
    """Can `child --relation--> anchor` actually place? -> (ok, suggestion, reason).

    Only object-to-object relations are judged (room relations always return ok).
    `suggestion` is a relation or "to:<factory-kind>" hint when a fix is obvious.
    """
    if relation == "ontop":
        if anchor in FLAT_TOP_ANCHORS:
            return True, None, ""
        if anchor in SHELF_ON_ANCHORS:
            return (False, "on",
                    f"{anchor} is open shelving (no flat top); `ontop` silently drops the "
                    f"object -- use `on` (its shelf SupportSurface) or re-anchor to a flat-top "
                    f"({', '.join(sorted(FLAT_TOP_ANCHORS))})")
        return (False, None,
                f"{anchor} exposes no flat top for `ontop`; anchor to a flat-top surface")
    if relation == "on":
        if anchor in SHELF_ON_ANCHORS or anchor in FLAT_TOP_ANCHORS:
            return True, None, ""
        return (False, "ontop" if anchor in FLAT_TOP_ANCHORS else None,
                f"{anchor} tags no SupportSurface for `on`")
    if relation == "front_against":
        if anchor in FRONT_AGAINST_ANCHORS:
            return True, None, ""
        if child in SEATING and anchor in DESK_LIKE:
            return (False, "front_to_front",
                    f"a seat at a wall-backed {anchor} must be `front_to_front` (chair front "
                    f"faces the desk front); `front_against` (child-front-vs-anchor-side) "
                    f"leaves the seat unplaceable. (A freestanding dining table/island is the "
                    f"exception -- chairs sit `front_against` its sides.)")
        return True, None, ""  # unknown pairing: let the solver try
    if relation == "front_to_front":
        return True, None, ""
    return True, None, ""  # side_by_side / back_to_back / etc.: not guarded here


@dataclass(frozen=True)
class RoomOntology:
    """Object/placement rules for a single room type."""

    room_type: str
    # fixtures that MAKE this room; presence is scored as the fraction of the
    # GROUND TRUTH's core fixtures the policy reproduced (not a fixed set).
    core_fixtures: frozenset[str]
    # always expected in a GT schema (used when there is no GT to compare to).
    always: frozenset[str]
    # room-appropriate extras: a small bonus when shared with the GT, absence
    # never penalized.
    optional_good: frozenset[str]
    # objects that never belong here -> flat forbidden penalty each.
    forbidden: frozenset[str]
    # relation classes
    wall_mount_only: frozenset[str]   # mirror/art/hardware: on_wall only
    ceiling_only: frozenset[str]      # ceiling lights: hanging only
    floor_wall_only: frozenset[str]   # stands on floor, back to a wall
    floor_any: frozenset[str]         # stands on floor, wall optional
    max_objects: int

    # ---- capitalized aliases so a shim can re-export as module attrs ------- #
    @property
    def CORE_FIXTURES(self) -> frozenset[str]: return self.core_fixtures
    @property
    def ALWAYS(self) -> frozenset[str]: return self.always
    @property
    def OPTIONAL_GOOD(self) -> frozenset[str]: return self.optional_good
    @property
    def FORBIDDEN(self) -> frozenset[str]: return self.forbidden
    @property
    def WALL_MOUNT_ONLY(self) -> frozenset[str]: return self.wall_mount_only
    @property
    def CEILING_ONLY(self) -> frozenset[str]: return self.ceiling_only
    @property
    def FLOOR_WALL_ONLY(self) -> frozenset[str]: return self.floor_wall_only
    @property
    def FLOOR_ANY(self) -> frozenset[str]: return self.floor_any
    @property
    def MAX_OBJECTS(self) -> int: return self.max_objects
    # shared sets, exposed on the instance for shim/back-compat convenience
    STORAGE = STORAGE
    FOOD_TABLEWARE = FOOD_TABLEWARE
    WALL_ALIGNED_RELATIONS = WALL_ALIGNED_RELATIONS
    ON_RELATIONS = ON_RELATIONS

    def allowed_room_relations(self, factory: str) -> frozenset[str] | None:
        """Room relations physically sensible for `factory`, or None if the
        factory is not an object this room has an opinion about."""
        if factory in self.wall_mount_only:
            return frozenset({"on_wall"})
        if factory in self.ceiling_only:
            return frozenset({"hanging"})
        if factory in self.floor_wall_only:
            return frozenset({"against_wall"})
        if factory in self.floor_any:
            return frozenset({"against_wall", "on_floor"})
        return None

    def relation_ok(self, factory: str, relation: str) -> bool:
        """True if `relation` is a sensible ROOM relation for `factory` (object-
        to-object relations are always accepted; they are judged by the parent)."""
        allowed = self.allowed_room_relations(factory)
        if allowed is None or relation in ON_RELATIONS or relation in _OBJ_ANCHORED:
            return True
        return relation in allowed

    def category(self, factory: str) -> str:
        """Coarse label used for logging / debugging."""
        if factory in self.core_fixtures:
            return "core"
        if factory in self.forbidden:
            return "forbidden"
        if factory in self.optional_good:
            return "optional_good"
        return "other"


# --------------------------------------------------------------------------- #
# Bathroom (copied verbatim from bathroom_ontology.py v2, MAX_OBJECTS=6)       #
# --------------------------------------------------------------------------- #
_BATH_FOOD = FOOD_TABLEWARE
_BATH_OTHER_FORBIDDEN = frozenset({
    "TableDiningFactory", "CoffeeTableFactory", "SideTableFactory", "SimpleDeskFactory",
    "ChairFactory", "OfficeChairFactory", "BarChairFactory",
    "SofaFactory", "ArmChairFactory", "BedFactory",
    "TVStandFactory", "KitchenIslandFactory",
    "OvenFactory", "MicrowaveFactory", "BeverageFridgeFactory",
})

BATHROOM = RoomOntology(
    room_type="Bathroom",
    core_fixtures=frozenset({"ToiletFactory", "StandingSinkFactory", "BathtubFactory"}),
    always=frozenset({"StandingSinkFactory"}),
    optional_good=frozenset({
        "LargeShelfFactory", "CellShelfFactory", "SimpleBookcaseFactory",
        "PlantContainerFactory", "VaseFactory",
        "MirrorFactory", "WallArtFactory", "RugFactory", "HardwareFactory",
        "CeilingLightFactory", "FloorLampFactory", "LampFactory", "DeskLampFactory",
    }),
    forbidden=_BATH_OTHER_FORBIDDEN | _BATH_FOOD,
    wall_mount_only=frozenset({"MirrorFactory", "WallArtFactory", "HardwareFactory"}),
    ceiling_only=frozenset({"CeilingLightFactory"}),
    floor_wall_only=frozenset({
        "ToiletFactory", "StandingSinkFactory",
        "LargeShelfFactory", "CellShelfFactory", "SimpleBookcaseFactory",
    }),
    floor_any=frozenset({
        "BathtubFactory", "PlantContainerFactory", "VaseFactory", "RugFactory",
        "FloorLampFactory", "LampFactory", "DeskLampFactory",
    }),
    max_objects=6,
)


# --------------------------------------------------------------------------- #
# Bedroom                                                                     #
# --------------------------------------------------------------------------- #
# A bed makes a bedroom. Nightstands (SideTable), a wardrobe/dresser (shelves/
# bookcase), a desk, seating (ArmChair), lamps, a rug, mirror/art on the wall,
# and a ceiling light are all appropriate. Bathroom fixtures, kitchen
# appliances, dining tables and loose tableware are out of place.
_BEDROOM_FORBIDDEN = frozenset({
    # bathroom fixtures
    "ToiletFactory", "BathtubFactory", "StandingSinkFactory",
    # kitchen
    "KitchenIslandFactory", "OvenFactory", "MicrowaveFactory", "BeverageFridgeFactory",
    # dining
    "TableDiningFactory", "BarChairFactory",
}) | FOOD_TABLEWARE

BEDROOM = RoomOntology(
    room_type="Bedroom",
    core_fixtures=frozenset({"BedFactory"}),
    always=frozenset({"BedFactory"}),
    optional_good=frozenset({
        # nightstands / desk / seating
        "SideTableFactory", "SimpleDeskFactory", "OfficeChairFactory", "ArmChairFactory",
        "CoffeeTableFactory", "TVStandFactory",
        # storage (wardrobe / dresser stand-ins)
        "LargeShelfFactory", "CellShelfFactory", "SimpleBookcaseFactory",
        # decor
        "MirrorFactory", "WallArtFactory", "RugFactory",
        "PlantContainerFactory", "VaseFactory",
        # lighting
        "CeilingLightFactory", "FloorLampFactory", "LampFactory", "DeskLampFactory",
    }),
    forbidden=_BEDROOM_FORBIDDEN,
    wall_mount_only=frozenset({"MirrorFactory", "WallArtFactory", "HardwareFactory"}),
    ceiling_only=frozenset({"CeilingLightFactory"}),
    floor_wall_only=frozenset({
        "BedFactory", "SideTableFactory", "SimpleDeskFactory", "TVStandFactory",
        "LargeShelfFactory", "CellShelfFactory", "SimpleBookcaseFactory",
    }),
    floor_any=frozenset({
        "ArmChairFactory", "OfficeChairFactory", "CoffeeTableFactory",
        "RugFactory", "PlantContainerFactory", "VaseFactory",
        "FloorLampFactory", "LampFactory", "DeskLampFactory",
    }),
    max_objects=8,
)


# --------------------------------------------------------------------------- #
# Living room                                                                 #
# --------------------------------------------------------------------------- #
# A sofa makes a living room. A coffee table, TV/stand, armchairs, side tables,
# shelving/bookcase, a rug, mirror/art on the wall, plants and lamps are all
# appropriate. Bathroom fixtures, kitchen appliances, a bed, a formal dining
# table, bar chairs and loose tableware are out of place. Living rooms hold the
# most furniture of any indoor type, so MAX_OBJECTS is the largest (10).
_LIVING_FORBIDDEN = frozenset({
    # bathroom fixtures
    "ToiletFactory", "BathtubFactory", "StandingSinkFactory",
    # kitchen
    "KitchenIslandFactory", "OvenFactory", "MicrowaveFactory", "BeverageFridgeFactory",
    # bedroom / dining
    "BedFactory", "TableDiningFactory", "BarChairFactory",
}) | FOOD_TABLEWARE

LIVINGROOM = RoomOntology(
    room_type="LivingRoom",
    core_fixtures=frozenset({"SofaFactory"}),
    always=frozenset({"SofaFactory"}),
    optional_good=frozenset({
        # seating / tables
        "CoffeeTableFactory", "ArmChairFactory", "ChairFactory", "SideTableFactory",
        "TVStandFactory", "SimpleDeskFactory", "OfficeChairFactory",
        # storage
        "LargeShelfFactory", "CellShelfFactory", "SimpleBookcaseFactory",
        # decor
        "MirrorFactory", "WallArtFactory", "RugFactory",
        "PlantContainerFactory", "VaseFactory",
        # lighting
        "CeilingLightFactory", "FloorLampFactory", "LampFactory", "DeskLampFactory",
    }),
    forbidden=_LIVING_FORBIDDEN,
    wall_mount_only=frozenset({"MirrorFactory", "WallArtFactory", "HardwareFactory"}),
    ceiling_only=frozenset({"CeilingLightFactory"}),
    # TV stands, shelving and desks stand with their back to a wall; a living-room
    # sofa commonly FLOATS facing the TV, so it is floor_any (against_wall optional).
    floor_wall_only=frozenset({
        "TVStandFactory", "SimpleDeskFactory",
        "LargeShelfFactory", "CellShelfFactory", "SimpleBookcaseFactory",
    }),
    floor_any=frozenset({
        "SofaFactory", "ArmChairFactory", "ChairFactory", "OfficeChairFactory",
        "CoffeeTableFactory", "SideTableFactory",
        "RugFactory", "PlantContainerFactory", "VaseFactory",
        "FloorLampFactory", "LampFactory", "DeskLampFactory",
    }),
    max_objects=10,
)


# --------------------------------------------------------------------------- #
# Kitchen                                                                     #
# --------------------------------------------------------------------------- #
# A stove/oven and a fridge define a kitchen. An island, a microwave, cabinets
# (open shelving / bookcase stand-ins), an eat-in dining table + chairs or
# barstools at the island, plus tableware, a rug, plants, wall art and ceiling
# lights are appropriate. Bathroom fixtures, a bed, a sofa/armchair/coffee table,
# a TV stand and a desk are out of place. Appliances and cabinets line the walls;
# the island sits free in the middle; barstools tuck against the island's side.
_KITCHEN_FORBIDDEN = frozenset({
    # bathroom fixtures
    "ToiletFactory", "BathtubFactory", "StandingSinkFactory",
    # bedroom
    "BedFactory",
    # living-room seating / tables
    "SofaFactory", "ArmChairFactory", "CoffeeTableFactory", "TVStandFactory",
    # office
    "SimpleDeskFactory", "OfficeChairFactory",
})

KITCHEN = RoomOntology(
    room_type="Kitchen",
    core_fixtures=frozenset({"OvenFactory", "BeverageFridgeFactory"}),
    always=frozenset({"OvenFactory", "BeverageFridgeFactory"}),
    optional_good=frozenset({
        # island / appliances
        "KitchenIslandFactory", "MicrowaveFactory",
        # eat-in seating
        "TableDiningFactory", "ChairFactory", "BarChairFactory",
        # cabinets / storage
        "LargeShelfFactory", "CellShelfFactory", "SimpleBookcaseFactory",
        # decor
        "WallArtFactory", "RugFactory", "PlantContainerFactory", "VaseFactory",
        # lighting
        "CeilingLightFactory", "LampFactory",
    }) | FOOD_TABLEWARE,
    forbidden=_KITCHEN_FORBIDDEN,
    wall_mount_only=frozenset({"WallArtFactory", "MirrorFactory", "HardwareFactory"}),
    ceiling_only=frozenset({"CeilingLightFactory"}),
    # appliances + cabinets stand with their back to a wall (the run of counters).
    floor_wall_only=frozenset({
        "OvenFactory", "BeverageFridgeFactory", "MicrowaveFactory",
        "LargeShelfFactory", "CellShelfFactory", "SimpleBookcaseFactory",
    }),
    # the island floats in the middle; an eat-in table, chairs, rug, plants free.
    floor_any=frozenset({
        "KitchenIslandFactory", "TableDiningFactory", "ChairFactory", "BarChairFactory",
        "RugFactory", "PlantContainerFactory", "VaseFactory", "LampFactory",
    }),
    max_objects=9,
)


# --------------------------------------------------------------------------- #
# Dining room                                                                 #
# --------------------------------------------------------------------------- #
# A dining table makes a dining room; chairs sit `front_against` its sides. A
# sideboard/credenza (TV stand stand-in) or a hutch/china cabinet (shelving), a
# rug under the table, a chandelier, wall art / a mirror, plants and tableware on
# the table are appropriate. Bathroom fixtures, a bed, kitchen appliances, a
# sofa/coffee table, a desk and barstools are out of place.
_DINING_FORBIDDEN = frozenset({
    # bathroom fixtures
    "ToiletFactory", "BathtubFactory", "StandingSinkFactory",
    # bedroom
    "BedFactory",
    # kitchen appliances / island / barstools
    "OvenFactory", "MicrowaveFactory", "BeverageFridgeFactory",
    "KitchenIslandFactory", "BarChairFactory",
    # living-room / office
    "SofaFactory", "CoffeeTableFactory", "SimpleDeskFactory", "OfficeChairFactory",
})

DININGROOM = RoomOntology(
    room_type="DiningRoom",
    core_fixtures=frozenset({"TableDiningFactory"}),
    always=frozenset({"TableDiningFactory"}),
    optional_good=frozenset({
        # seating
        "ChairFactory", "ArmChairFactory",
        # sideboard / hutch / china cabinet
        "TVStandFactory", "LargeShelfFactory", "CellShelfFactory", "SimpleBookcaseFactory",
        # decor
        "MirrorFactory", "WallArtFactory", "RugFactory",
        "PlantContainerFactory", "VaseFactory",
        # lighting
        "CeilingLightFactory", "FloorLampFactory", "LampFactory",
    }) | FOOD_TABLEWARE,
    forbidden=_DINING_FORBIDDEN,
    wall_mount_only=frozenset({"MirrorFactory", "WallArtFactory", "HardwareFactory"}),
    ceiling_only=frozenset({"CeilingLightFactory"}),
    # a sideboard/credenza and a hutch/china cabinet stand with their back to a wall.
    floor_wall_only=frozenset({
        "TVStandFactory", "LargeShelfFactory", "CellShelfFactory", "SimpleBookcaseFactory",
    }),
    # the dining table sits free; chairs, rug, plants, floor lamp free.
    floor_any=frozenset({
        "TableDiningFactory", "ChairFactory", "ArmChairFactory",
        "RugFactory", "PlantContainerFactory", "VaseFactory",
        "FloorLampFactory", "LampFactory",
    }),
    max_objects=9,
)


# --------------------------------------------------------------------------- #
# registry                                                                    #
# --------------------------------------------------------------------------- #
ONTOLOGY: dict[str, RoomOntology] = {
    "Bathroom": BATHROOM,
    "Bedroom": BEDROOM,
    "LivingRoom": LIVINGROOM,
    "Kitchen": KITCHEN,
    "DiningRoom": DININGROOM,
}


def get(room_type: str) -> RoomOntology | None:
    """Ontology for a room type (exact, case-sensitive), or None if unknown."""
    return ONTOLOGY.get(room_type)


# Neutral fallback for an unrecognized room type: no forbidden objects, a
# permissive object budget, and relation checks that pass anything. Lets the
# reward degrade to set/placement grading instead of returning 0.
FALLBACK = RoomOntology(
    room_type="__fallback__",
    core_fixtures=frozenset(),
    always=frozenset(),
    optional_good=frozenset(),
    forbidden=frozenset(),
    wall_mount_only=frozenset(),
    ceiling_only=frozenset(),
    floor_wall_only=frozenset(),
    floor_any=frozenset(),
    max_objects=10,
)


def get_or_fallback(room_type: str | None) -> RoomOntology:
    """Ontology for a room type, or the neutral FALLBACK if unknown/None."""
    if room_type is None:
        return FALLBACK
    return ONTOLOGY.get(room_type, FALLBACK)
