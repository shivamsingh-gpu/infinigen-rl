"""Render an IndoorConfig with Infinigen's indoor pipeline, or a stub.

The policy's relational spec is written to `<out_dir>/spec.json` and picked up
by Infinigen via the INFINIGEN_RL_SPEC env var (the `rl_inject` hook in
`home.py` compiles it into constraint-DSL). Indoor generation is two stages --
`coarse` (room layout + relational solve) then `render` -- with no
populate/fine_terrain stage (populate happens inside coarse).

GPU/CPU pinning and the Infinigen root/venv resolution are shared with the
nature renderer (`render_scene`), so a fix there flows through to both.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

from config_space import SceneConfig  # noqa: F401  (kept so both renderers share a module namespace)
from indoor_config_space import IndoorConfig
from render_scene import INFINIGEN_PY, INFINIGEN_ROOT, _affinity

# Fast preview by default: 200 Cycles samples instead of the 8192 indoor
# default (~25s/frame vs ~10min). Set RL_INDOOR_SAMPLES to override.
NUM_SAMPLES = int(os.environ.get("RL_INDOOR_SAMPLES", "200"))
MIN_SAMPLES = int(os.environ.get("RL_INDOOR_MIN_SAMPLES", "32"))
# Reward camera: a fixed elevated-corner pose that frames the whole room (all
# fixtures) deterministically, so reward reflects placement quality rather than
# which wall a random pose faced. Implemented in generate_indoors.pose_cameras
# (gated on RL_INDOOR_CAMERA_CORNER). Set RL_INDOOR_CAMERA_CORNER=0 to fall back
# to Infinigen's random pose search.
CAMERA_CORNER = os.environ.get("RL_INDOOR_CAMERA_CORNER", "1") not in ("0", "", "false", "False")

def _env(*names: str, default: str = "") -> str:
    """First set (non-empty) value among env var `names`, else `default`.

    Lets a generic RL_INDOOR_ROOM_* knob be read while keeping the historical
    RL_INDOOR_BATH_* names working as aliases."""
    for n in names:
        v = os.environ.get(n)
        if v not in (None, ""):
            return v
    return default


# Bathroom footprint knobs forwarded to home_room_constraints (gin). Only the
# Bathroom room type has these kwargs on home_room_constraints; other rooms rely
# on the predefined floor plan for sizing. The stock soft target is 8..~12 m^2
# with a 2 m short-side threshold, which rendered hall-sized bathrooms.
BATH_HINGE = os.environ.get("RL_INDOOR_BATH_HINGE", "0.35")
BATH_NARROW = os.environ.get("RL_INDOOR_BATH_NARROW", "1.7")
# weight on log(aspect ratio) for Bathroom rooms (stock 40 let a 10 m corridor
# pass as a bathroom in the v2 smoke); higher = squarer rooms.
BATH_ASPECT_W = os.environ.get("RL_INDOOR_BATH_ASPECT_W", "400")

_FLOOR_RELATIONS = {"against_wall", "on_floor"}

# Explicit area override (m^2) for ANY room; "" = derive from the spec.
AREA_OVERRIDE = _env("RL_INDOOR_ROOM_AREA", "RL_INDOOR_BATH_AREA", default="")

# Deterministic single-room floor plan (Infinigen's PredefinedFloorPlanSolver via
# gin `Solver.floor_plan=<json>`). The stock floor-plan annealer treats the room
# area only as a weak soft target: with fast_solve.gin it still produced 2x2 m
# (tub unplaceable) and 4.5x5 m bathrooms in the v2 smoke. Set
# RL_INDOOR_PREDEFINED=0 to fall back to the annealer + gin area knobs.
PREDEFINED = _env("RL_INDOOR_PREDEFINED", "RL_INDOOR_BATH_PREDEFINED", default="1") \
    not in ("0", "", "false")
# Daylight: stock sun_elevation is clip_gaussian(40, 25, 6, 70) deg -> some seeds
# render as night (luma ~30). Pin it for GT validation; "" = stock random.
SUN_ELEVATION = os.environ.get("RL_INDOOR_SUN_ELEVATION", "45")

# Per-room-type footprint model used to size the predefined floor plan. `base`
# m^2 grows by `per_extra` for each floor-standing object beyond `extra_after`
# (and by `big_bonus` if a `big_item` factory is present), clamped to
# [clamp_lo, clamp_hi]. `min_side`/`aspect` shape the rectangle. Bathroom values
# are unchanged from the validated v2 tuning; Bedroom is larger (a bed + a couple
# of nightstands + wardrobe reads wrong in a 6 m^2 box).
# `cam_eye_h` is the corner reward camera's eye height as a fraction of room
# height (generate_indoors RL_INDOOR_CAMERA_CORNER_HEIGHT). Small bathrooms
# frame fine from a high 0.85 eye looking down; a larger bedroom needs a lower,
# near-eye-level camera (~0.5) or the low bed dominates a steep bird's-eye view.
ROOM_SIZING: dict[str, dict] = {
    "Bathroom": dict(base=5.5, big_item="BathtubFactory", big_bonus=2.5,
                     per_extra=1.0, extra_after=3, clamp_lo=0.0, clamp_hi=10.5,
                     min_side=2.5, aspect=1.25, cam_eye_h=0.85, exposure=None),
    # cam_eye_h 0.68: a bed has a large footprint so the corner-camera aim is
    # pulled onto it; at 0.5 (eye ~1.45 m) the camera stares into the duvet
    # (235 smoke). ~0.68 looks *down over* the bed and shows the whole room.
    # exposure 4.2 (> stock 3): dark-material scenes (253 industrial loft) render
    # near-black at stock; a modest bump keeps the well-lit seeds acceptable.
    # base/clamp_hi/min_side sized up from the iter-2 smoke: in a 4x4 m room no
    # corner cleared min_clear=0.5 (235 fell back to a steep bird's-eye that the
    # bed filled) and a chair front_against a desk failed to place (05). ~4.5 m
    # sides give the corner camera a real pose and the solver room to place.
    "Bedroom": dict(base=14.0, big_item=None, big_bonus=0.0,
                    per_extra=1.5, extra_after=4, clamp_lo=11.0, clamp_hi=22.0,
                    min_side=3.2, aspect=1.2, cam_eye_h=0.68, exposure=4.2),
    # Living rooms are the largest indoor type: a sofa + coffee table + TV/stand +
    # a seating cluster + shelving need real floor. base/clamp sized above bedroom;
    # wider aspect (1.35) matches the typical long living-room rectangle. cam_eye_h
    # 0.70 looks down over low sofa/coffee-table furniture (same reasoning as the
    # bed). exposure 4.2 guards the moody/dark-material scenes as in bedroom.
    "LivingRoom": dict(base=18.0, big_item="SofaFactory", big_bonus=2.0,
                       per_extra=1.5, extra_after=5, clamp_lo=14.0, clamp_hi=30.0,
                       min_side=3.8, aspect=1.35, cam_eye_h=0.70, exposure=4.2),
    # Kitchen: an island (freestanding, needs 0.7-3 m clearance to the counters per
    # home.py) + a run of wall cabinets/appliances + maybe an eat-in table needs
    # real floor. cam_eye_h 0.75: the island/counter tops sit ~0.9 m, so a slightly
    # higher eye looks down over them (same reasoning as the bed/sofa). exposure 4.2
    # guards dark-material (matte-black/wood) kitchens.
    "Kitchen": dict(base=15.0, big_item="KitchenIslandFactory", big_bonus=2.0,
                    per_extra=1.3, extra_after=5, clamp_lo=12.0, clamp_hi=26.0,
                    min_side=3.5, aspect=1.30, cam_eye_h=0.75, exposure=4.2),
    # Dining room: a table with chairs around all four sides needs clearance to pull
    # them out; a sideboard/hutch lines a wall. cam_eye_h 0.72 looks down over the
    # table + chairs. exposure 4.2 as above.
    "DiningRoom": dict(base=14.0, big_item="TableDiningFactory", big_bonus=2.0,
                       per_extra=1.3, extra_after=5, clamp_lo=12.0, clamp_hi=24.0,
                       min_side=3.4, aspect=1.25, cam_eye_h=0.72, exposure=4.2),
}
_DEFAULT_SIZING = dict(base=9.0, big_item=None, big_bonus=0.0,
                       per_extra=1.2, extra_after=4, clamp_lo=0.0, clamp_hi=18.0,
                       min_side=2.5, aspect=1.2, cam_eye_h=0.6, exposure=None)

# Infinigen's PredefinedFloorPlanSolver derives the room's Semantics from the
# key via `Semantics(name.split("_")[0])` (room/base.py `room_type`), which
# compares against the enum VALUE, not our CamelCase room_type. Most values are
# the lowercased name (bathroom/bedroom/kitchen) so `room_type.lower()` happened
# to work, but the compound rooms use a hyphen ("living-room"/"dining-room").
# Map explicitly so the key token is a valid Semantics value.
_SEMANTICS_VALUE: dict[str, str] = {
    "Bathroom": "bathroom",
    "Bedroom": "bedroom",
    "Kitchen": "kitchen",
    "LivingRoom": "living-room",
    "DiningRoom": "dining-room",
}


def _semantics_value(room_type: str) -> str:
    return _SEMANTICS_VALUE.get(room_type, room_type.lower())


def _sizing(cfg: IndoorConfig) -> dict:
    return ROOM_SIZING.get(cfg.room_type, _DEFAULT_SIZING)


def area_for(cfg: IndoorConfig) -> float:
    """Lower edge of the room area zero-band (m^2) for this spec.

    Compact by default, growing with the number of floor-standing objects so a
    fully-furnished room is not squeezed too small (the bathroom v2 smoke lost
    the tub in a 4 m^2 room). Env RL_INDOOR_ROOM_AREA (or RL_INDOOR_BATH_AREA)
    overrides for any room."""
    if AREA_OVERRIDE:
        return float(AREA_OVERRIDE)
    s = _sizing(cfg)
    n_floor = sum(max(1, o.count) for o in cfg.objects if o.relation in _FLOOR_RELATIONS)
    has_big = bool(s["big_item"]) and any(o.factory == s["big_item"] for o in cfg.objects)
    area = s["base"] + (s["big_bonus"] if has_big else 0.0) \
        + s["per_extra"] * max(0, n_floor - s["extra_after"])
    return round(max(s["clamp_lo"], min(area, s["clamp_hi"])), 2)


# back-compat alias (older callers / logs used bath_area_for)
bath_area_for = area_for


def _min_side(cfg: IndoorConfig) -> float:
    return float(_env("RL_INDOOR_ROOM_MIN_SIDE", "RL_INDOOR_BATH_MIN_SIDE",
                      default=str(_sizing(cfg)["min_side"])))


def floor_plan(cfg: IndoorConfig, out_dir: Path) -> Path:
    """Write a one-room floor plan JSON sized from the spec for cfg.room_type.

    Rectangle w x h on a 0.5 m grid (per-room aspect, sides >= min_side) whose
    area is the spec's target; door centred on the y=0 wall, window centred on
    the opposite wall. Room key is ``<semantics_value>_0/0`` (the key
    PredefinedFloorPlanSolver expects). Format = floor_plans/predefined.json.
    """
    area = area_for(cfg)
    ms = _min_side(cfg)
    aspect = _sizing(cfg)["aspect"]
    w = max(ms, round(math.sqrt(area * aspect) * 2) / 2)
    h = max(ms, round((area / w) * 2) / 2)
    room_key = f"{_semantics_value(cfg.room_type)}_0/0"
    plan = {
        "rooms": {room_key: {"shape": f"shapely.box(0,0,{w},{h})"}},
        "doors": {"door": {"shape": f"shapely.LineString([({w/2-0.45},0),({w/2+0.45},0)])"}},
        "opens": {},
        "interiors": {},
        "windows": {"window": {"shape": f"shapely.LineString([({w/2-0.6},{h}),({w/2+0.6},{h})])"}},
    }
    path = out_dir / "floor_plan.json"
    path.write_text(json.dumps(plan, indent=1))
    return path


# back-compat alias
bath_floor_plan = floor_plan

_ROOM_PALETTE = {
    "DiningRoom": (170, 140, 110),
    "LivingRoom": (150, 150, 165),
    "Bedroom": (140, 120, 150),
    "Kitchen": (180, 180, 170),
    "Bathroom": (170, 195, 205),
}


def render_stub(cfg: IndoorConfig, out_dir: Path, size: int = 512) -> Path:
    """Cheap placeholder render -- a colored card labeled with the room spec."""
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (size, size), _ROOM_PALETTE.get(cfg.room_type, (128, 128, 128)))
    draw = ImageDraw.Draw(img)
    lines = [cfg.room_type] + [
        f"{o.count}x {o.factory} [{o.relation}"
        + (f"->{o.target}]" if o.target else "]")
        for o in cfg.objects[:10]
    ]
    draw.text((10, 10), "\n".join(lines), fill=(20, 20, 20))
    out = out_dir / "render.png"
    img.save(out)
    return out


def _run_stage(task: str, gin: list[str], params: list[str],
               seed_hex: str, in_dir: Path | None, out_dir: Path,
               slot: int, spec_path: Path, extra_env: dict | None = None) -> None:
    gpu, cpulist, n_threads = _affinity(slot)

    cmd = [
        str(INFINIGEN_PY), "-m", "infinigen_examples.generate_indoors",
        "--seed", seed_hex,
        "--task", *task.split(),
        "-g", *gin,
        "--output_folder", str(out_dir),
    ]
    if params:
        cmd += ["-p", *params]
    if in_dir is not None:
        cmd += ["--input_folder", str(in_dir)]

    if cpulist and shutil.which("taskset"):
        cmd = ["taskset", "-c", cpulist] + cmd

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["INFINIGEN_RL_SPEC"] = str(spec_path)  # the hook reads this
    env["RL_INDOOR_CAMERA_CORNER"] = "1" if CAMERA_CORNER else "0"
    if os.environ.get("RL_INDOOR_EXCLUSIVE"):
        # policy "owns" the room: rl_inject drops all stock furniture rules so
        # only the policy's placements populate the scene (clean reward credit).
        env["INFINIGEN_RL_EXCLUSIVE"] = "1"
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "TBB_NUM_THREADS"):
        env[var] = str(n_threads)
    # per-room camera tuning etc.; never clobber a value the user set at submit
    for k, v in (extra_env or {}).items():
        env.setdefault(k, v)

    # Per-scene, per-stage log (the sbatch stdout interleaves 8 workers, which
    # made the [rl-camera] / [rl] lines unattributable). Tail it on failure.
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir.parent / f"{task.split()[0]}.log"
    with open(log_path, "w") as log:
        log.write("+ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=INFINIGEN_ROOT, env=env,
                              stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        tail = "".join(open(log_path, errors="replace").readlines()[-40:])
        raise RuntimeError(f"{task} stage failed (rc={proc.returncode}); log={log_path}\n{tail}")


def render_infinigen(cfg: IndoorConfig, out_dir: Path, slot: int = 0) -> Path:
    """coarse (layout + relational solve, incl. the policy's spec) -> render."""
    cfg = cfg.sanitize()
    out_dir.mkdir(parents=True, exist_ok=True)

    spec_path = out_dir / "spec.json"
    spec_path.write_text(cfg.to_json())

    coarse = out_dir / "coarse"
    frames = out_dir / "frames"

    seed_hex = format(int(cfg.seed) % (2 ** 32), "x")
    gin = ["fast_solve.gin", "singleroom.gin"]

    # Camera pose is set during coarse (pose_cameras): with RL_INDOOR_CAMERA_CORNER
    # (default on, forwarded via env in _run_stage) the search is bypassed for a
    # deterministic full-room corner pose. No extra gin needed.
    coarse_params = [
        "compose_indoors.terrain_enabled=False",
        f'restrict_solving.restrict_parent_rooms=["{cfg.room_type}"]',
    ]
    # Deterministic sizing: a predefined single-room floor plan (room-agnostic).
    if PREDEFINED:
        fp = floor_plan(cfg, out_dir)
        coarse_params.append(f'Solver.floor_plan="{fp}"')
    # Bathroom is the only room type with area/aspect kwargs on
    # home_room_constraints; they refine the soft area score even under the
    # predefined plan. Other rooms rely on the predefined plan for sizing.
    if cfg.room_type == "Bathroom":
        coarse_params += [
            f"home_room_constraints.bathroom_area={area_for(cfg)}",
            f"home_room_constraints.bathroom_area_hinge={BATH_HINGE}",
            f"home_room_constraints.bathroom_narrowness={BATH_NARROW}",
            f"home_room_constraints.bathroom_aspect_weight={BATH_ASPECT_W}",
        ]
    if SUN_ELEVATION:
        coarse_params.append(f"nishita_lighting.sun_elevation={SUN_ELEVATION}")
    if os.environ.get("RL_INDOOR_DOORS_CLOSED", "1") not in ("0", "", "false"):
        # open door leaves swung into the doorway camera's view (124/139/186/191)
        coarse_params.append("populate_doors.all_closed=True")
    if os.environ.get("RL_INDOOR_NO_SHUTTERS", "1") not in ("0", "", "false"):
        # window shutters swing open into the room and block the camera (124)
        coarse_params.append("populate_windows.no_shutter=True")
    # more annealing for the floor+wall stage (fast_solve.gin = 100): small rooms
    # with 4 fixtures need it to find a non-colliding layout
    coarse_params.append(f"compose_indoors.solve_steps_large={os.environ.get('RL_INDOOR_SOLVE_STEPS_LARGE', '250')}")
    # Per-room corner-camera eye height (fraction of room height); larger rooms
    # want a lower, near-eye-level camera. setdefault in _run_stage keeps any
    # RL_INDOOR_CAMERA_CORNER_HEIGHT the user set at submit.
    cam_env = {"RL_INDOOR_CAMERA_CORNER_HEIGHT": str(_sizing(cfg)["cam_eye_h"])}
    _run_stage(
        "coarse", gin, coarse_params,
        seed_hex, None, coarse, slot, spec_path, extra_env=cam_env,
    )
    render_params = [
        f"configure_render_cycles.num_samples={NUM_SAMPLES}",
        f"configure_render_cycles.min_samples={MIN_SAMPLES}",
    ]
    # exposure: submit-time RL_INDOOR_EXPOSURE wins; else the per-room default
    # (stock base_indoors.gin = 3; None keeps stock, e.g. Bathroom).
    exposure = os.environ.get("RL_INDOOR_EXPOSURE") or _sizing(cfg).get("exposure")
    if exposure:
        render_params.append(f"configure_render_cycles.exposure={exposure}")
    if SUN_ELEVATION:
        render_params.append(f"nishita_lighting.sun_elevation={SUN_ELEVATION}")
    _run_stage(
        "render", gin, render_params,
        seed_hex, coarse, frames, slot, spec_path,
    )

    rgb = frames / "Image" / "camera_0"
    pngs = sorted(rgb.glob("Image_*.png"))
    if not pngs:
        raise RuntimeError(f"no RGB frame under {rgb}")
    return pngs[0]


def render(cfg: IndoorConfig, out_dir: Path, stub: bool = False, slot: int = 0) -> Path:
    return render_stub(cfg, out_dir) if stub else render_infinigen(cfg, out_dir, slot)
