# Infinigen core patches

The RL integration touches **three** files in the infinigen checkout (`$INFINIGEN_ROOT`). All
changes are **env-gated**: with `INFINIGEN_RL_SPEC` / `RL_INDOOR_*` unset, stock Infinigen behaves
exactly as upstream. Apply these to a fresh infinigen checkout before rendering generated layouts
or running the RL hook.

Everything else lives in the scaffold (`infinigen/rl_infinigen_beginner/`) as **new files** and is
not a core edit.

---

## 1. `src/infinigen_examples/constraints/rl_inject.py` — NEW FILE (the whole hook)
Copy `patches/infinigen/rl_inject.py` into that path. It compiles a JSON scene spec (read from the
`INFINIGEN_RL_SPEC` env var) into Infinigen's constraint DSL. Responsibilities:
- No-op unless `INFINIGEN_RL_SPEC` is set (stock + `INFINIGEN_DEMO_BOX` demo unaffected).
- Relation compilation: `against_wall` → `on_floor` AND `against_wall` (fixes the v1 floating-tub
  bug), `on_wall` → `flush_wall` + injected height band, `on_floor`, `hanging`, object-to-object
  (`ontop`/`on`/`front_against`/`side_by_side`).
- Labeled `id`s become distinct factory subclasses (so two same-type objects can carry different
  constraints).
- `_ARRANGE` table compiles the soft `arrange` objectives (distance/spacing/focus/align/symmetry/center).
- `_SEAT_PROXIMITY` seat-placement fix; exclusive-mode (`INFINIGEN_RL_EXCLUSIVE`) strips stock
  furniture rules so only the policy's objects populate the room.

### The `spacing` render-crash fix (already in the bundled copy)
In `_ARRANGE`, `spacing` maps to `cl.distance`, **not** `cl.min_dist_2d`:
```python
# BROKEN (upstream min_dist_2d_impl passes `b` as a list of names, but
# trimesh_geometry.min_dist_2d() calls b.to_planar() directly -> AttributeError
# on any object SET, crashing the coarse solve):
#   "spacing": (cl.min_dist_2d, True, "max"),
# FIX (cl.distance -> min_dist() supports many-to-many name lists; maximizing
# pulls the two sets apart = same clearance effect). Render-only: the schema
# keeps the word "spacing" and the schema-vs-schema reward is unchanged.
"spacing": (cl.distance, True, "max"),
```

---

## 2. `src/infinigen_examples/constraints/home.py` — 3 additive lines
```python
# (a) top-of-file import
from .rl_inject import maybe_inject_rl_constraints, register_rl_aliases  # RL: compile+inject JSON spec

# (b) in the room-constraint builder: register one alias factory per labeled object
#     (guarded; see register_rl_aliases)

# (c) at the end of the constraint assembly (where `constraints`, `score_terms`,
#     `rooms`, `obj` are in scope):
    # RL (opt-in via env INFINIGEN_RL_SPEC=/path/to/spec.json): compile the
    # policy's JSON scene spec into constraint-DSL and add it here. No-op when
    # the env var is unset, so normal runs are unaffected.
    maybe_inject_rl_constraints(constraints, score_terms, rooms, obj)
```

---

## 3. `src/infinigen_examples/generate_indoors.py` — deterministic reward camera
Adds `_rl_corner_camera_enabled()`, `_rl_corner_camera_pose()`, and pan-camera helpers, and
modifies `pose_cameras()` to **bypass the stochastic pose search** when `RL_INDOOR_CAMERA_CORNER=1`,
placing a fixed elevated-corner pose that frames the whole room (so every render is a deterministic,
comparable viewpoint). Governed by `RL_INDOOR_CAMERA_CORNER*` env knobs
(`_INSET`, `_HEIGHT`, `_AIM`, `_LENS`, …) and `RL_INDOOR_EXCLUSIVE` for exclusive population.

The hook region in `pose_cameras()`:
```python
    def pose_cameras():
        n_pan = _pan_camera_frames()
        if n_pan > 0:                      # turntable pan (optional)
            _setup_pan_camera(solved_rooms, camera_rigs, n_pan); return [], None
        if _rl_corner_camera_enabled():    # RL fixed corner reward camera
            ... compute (loc, rot) via _rl_corner_camera_pose(...) ...
            for rig in camera_rigs:
                rig.location = loc; rig.rotation_euler = rot
            return ...
        ... stock selection-ratio pose search ...
```

---

## Relevant env vars (set at render time)
| var | effect |
|---|---|
| `INFINIGEN_RL_SPEC=/path/spec.json` | activates the hook; compiles that schema |
| `INFINIGEN_RL_EXCLUSIVE=1` | only the policy's objects populate the room |
| `RL_INDOOR_CAMERA_CORNER=1` | deterministic corner reward camera (default on in the drivers) |
| `RL_INDOOR_SAMPLES=200` | Blender render samples (the 8192→200 speed fix) |
| `RL_SEAT_PROXIMITY_W=8` | weight of the seat→table proximity pull |

`src/render_indoor.py` sets these for you per scene; you normally don't set them by hand.
