"""Render the bathroom ground-truth schemas with Infinigen and build a
reference-vs-render contact sheet + an automatic report for human visual
validation (Phase B).

Uses the fixed eye-level corner reward camera (RL_INDOOR_CAMERA_CORNER=1, the
default) so every GT renders from a deterministic full-room viewpoint, and
RL_INDOOR_EXCLUSIVE=1 so only the schema's own placements populate the room
(no stock furniture) -- what you see is exactly what the schema encodes.

Renders run in a small pool (one in-flight job per GPU slot). Output:
  gt_validation/bathroom/<stem>/                 (raw Infinigen output + coarse.log/render.log)
  gt_validation/bathroom/contact/<stem>.png      (reference | render side-by-side)
  gt_validation/bathroom/contact/_grid_<k>.png   (montage pages of the contact sheets)
  gt_validation/bathroom/report.json             (per-stem automatic checks, see _analyze)

Automatic FAIL conditions (still eyeball everything that passes):
  * render missing / crashed
  * mean luma < LUMA_MIN (black frame: camera inside geometry or unlit room)
  * a CORE fixture never got placed (count mismatch in solve_state); a missing
    optional object is only a WARN (still a plausible GT reference)
  * a floor fixture lacks a floor-contact relation in solve_state (floating)

Run (on a GPU node / via render_gt.sbatch):
  RL_INDOOR_EXCLUSIVE=1 python validate_gt_render.py --workers 8
  python validate_gt_render.py --only 02_cozy_bathroom 78_japanese_bathroom
  python validate_gt_render.py --report-only          # re-analyze existing output
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import re
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageDraw, ImageStat

import indoor_ontology as onto_reg
from indoor_config_space import IndoorConfig
from render_indoor import render, ROOM_SIZING, _DEFAULT_SIZING

BASE = Path(__file__).resolve().parents[1]

# Room selection (env ROOM=bedroom, default bathroom). Bathroom refs live in
# ext_indoor_bath; other rooms in ext_indoor_<room>.
ROOM = os.environ.get("ROOM", "bathroom").lower()
_ROOM_TYPE = {"bathroom": "Bathroom", "bedroom": "Bedroom"}
_REF_SUFFIX = {"bathroom": "bath", "bedroom": "bedroom"}
ROOM_TYPE = _ROOM_TYPE.get(ROOM, ROOM.capitalize())
onto = onto_reg.get_or_fallback(ROOM_TYPE)

GT_DIR = BASE / "gt_schemas" / ROOM
REF_DIR = BASE / "references" / f"ext_indoor_{_REF_SUFFIX.get(ROOM, ROOM)}"
OUT_DIR = BASE / "gt_validation" / ROOM
CONTACT_DIR = OUT_DIR / "contact"
REPORT = OUT_DIR / "report.json"

# "large room" warn threshold, per room type (bedrooms are legitimately bigger).
_SIZING = ROOM_SIZING.get(ROOM_TYPE, _DEFAULT_SIZING)
LARGE_AREA = float(os.environ.get("RL_GT_LARGE_AREA", str(_SIZING["clamp_hi"] + 1.5)))

LUMA_MIN = float(os.environ.get("RL_GT_LUMA_MIN", "15"))
GRID_COLS, GRID_ROWS, GRID_W = 2, 6, 900  # contact sheets per montage page


def _contact(stem: str, render_png: Path, h: int = 512) -> Path:
    """reference (left) | GT render (right), same height, labeled."""
    CONTACT_DIR.mkdir(parents=True, exist_ok=True)
    ref = Image.open(REF_DIR / f"{stem}.png").convert("RGB")
    ren = Image.open(render_png).convert("RGB")

    def _fit(im: Image.Image) -> Image.Image:
        w = int(im.width * h / im.height)
        return im.resize((w, h))

    ref, ren = _fit(ref), _fit(ren)
    pad = 8
    canvas = Image.new("RGB", (ref.width + ren.width + pad, h + 24), (255, 255, 255))
    canvas.paste(ref, (0, 24))
    canvas.paste(ren, (ref.width + pad, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 6), f"{stem}   |   REFERENCE (left)  vs  GT RENDER (right)", fill=(0, 0, 0))
    out = CONTACT_DIR / f"{stem}.png"
    canvas.save(out)
    return out


def _render_png(stem: str) -> Path | None:
    pngs = sorted((OUT_DIR / stem / "frames" / "Image" / "camera_0").glob("Image_*.png"))
    return pngs[0] if pngs else None


_CAM_RE = re.compile(
    r"\[rl-camera\] floor centroid=\(([-\d.]+), ([-\d.]+), [-\d.]+\) corner=\(([-\d.]+), ([-\d.]+)\)"
    r".*?(?:clearance=([-\d.]+))?"
)


def _analyze(stem: str) -> dict:
    """Automatic checks for one stem from its on-disk artifacts."""
    rep: dict = {"stem": stem, "fails": [], "warns": []}
    cfg = IndoorConfig.from_json((GT_DIR / f"{stem}.json").read_text()).sanitize()
    rep["n_spec_objects"] = sum(max(1, o.count) for o in cfg.objects)

    png = _render_png(stem)
    if png is None:
        rep["fails"].append("no render")
    else:
        luma, dfrac = _darkness(png)
        rep["luma"] = round(luma, 1)
        rep["dark_frac"] = round(dfrac, 2)
        if luma < LUMA_MIN:
            rep["fails"].append(f"black frame (luma {rep['luma']} < {LUMA_MIN})")
        elif luma < LUMA_DARK or dfrac > DARK_FRAC:
            rep["warns"].append(f"dark (luma {luma:.0f}, {dfrac:.0%} px < {DARK_PX})")

    # camera + room footprint from the per-scene coarse log
    log = OUT_DIR / stem / "coarse.log"
    if log.exists():
        txt = log.read_text(errors="replace")
        m = re.search(r"room floor area=([-\d.]+) m2 bbox=([-\d.]+)x([-\d.]+)", txt)
        if m:
            _a, w, h = map(float, m.groups())  # mesh area double-counts faces; use bbox
            a = w * h
            rep["room_wh"] = [round(w, 1), round(h, 1)]
            rep["room_area"] = round(a, 1)
            if a > LARGE_AREA:
                rep["warns"].append(f"room {w:.1f}x{h:.1f}={a:.0f} m2 (large)")
            if max(w, h) / max(1e-6, min(w, h)) > 2.2:
                rep["warns"].append(f"room elongated {w:.1f}x{h:.1f}")
        m = re.search(r"over_floor=(\d+)", txt)
        if m and int(m.group(1)) == 0:
            rep["warns"].append("no camera candidate over the room floor")
        m = re.search(r"clearance=([-\d.]+)", txt)
        if m:
            rep["cam_clearance"] = float(m.group(1))
            if rep["cam_clearance"] < 0.2:
                rep["warns"].append(f"camera clearance {rep['cam_clearance']:.2f} m")
        m = re.search(r"down_hit=(\w+)", txt)
        if m and m.group(1) != "True":
            rep["fails"].append("camera not over floor (down raycast miss)")
        if "Traceback" in txt:
            rep["warns"].append("traceback in coarse.log")

    # placement sanity from the solver state: every labeled object placed, and
    # floor fixtures have a floor-contact relation (not just a wall one).
    ss = OUT_DIR / stem / "coarse" / "solve_state.json"
    if ss.exists():
        objs = json.loads(ss.read_text()).get("objs", {})
        placed: dict[str, list[dict]] = {}
        for k, o in objs.items():
            gen = (o.get("generator") or "").split("(")[0]
            placed.setdefault(gen, []).append(o)
        rep["placed"] = {g: len(v) for g, v in placed.items()}
        for o in cfg.objects:
            key = o.id or o.factory
            got = placed.get(key, [])
            if len(got) < o.count:
                # A missing CORE fixture (bed; bath sink/toilet/tub) breaks the
                # scene -> FAIL. A missing optional object (e.g. a chair the
                # solver could not fit front_against a desk) still leaves a
                # plausible GT reference -> WARN, so the batch is not gated on
                # cosmetic solver misses. (Reward scores schema-vs-schema, not
                # the render, so an under-placed GT render is harmless.)
                msg = f"{key}: placed {len(got)}/{o.count}"
                bucket = "fails" if o.factory in onto.CORE_FIXTURES else "warns"
                rep[bucket].append(msg)
                continue
            if o.relation == "against_wall" or (
                o.relation == "on_floor" and o.factory in onto.FLOOR_WALL_ONLY | onto.FLOOR_ANY
            ):
                for inst in got:
                    tags = [
                        t for r in inst.get("relations", [])
                        for t in r["relation"].get("parent_tags", [])
                    ]
                    if "Subpart(support)" not in tags:
                        rep["fails"].append(f"{key}: no floor contact (floating)")
                        break
    seed_file = OUT_DIR / stem / "seed_used.txt"
    if seed_file.exists() and "attempt 0" not in seed_file.read_text():
        rep["warns"].append(f"rendered with retry seed {seed_file.read_text().strip()}")
    rep["status"] = "FAIL" if rep["fails"] else "ok"
    return rep


def _grid(stems: list[str]) -> list[Path]:
    """Tile the contact sheets into montage pages for quick scanning."""
    sheets = [CONTACT_DIR / f"{s}.png" for s in stems if (CONTACT_DIR / f"{s}.png").exists()]
    per = GRID_COLS * GRID_ROWS
    outs = []
    for k in range(math.ceil(len(sheets) / per)):
        chunk = sheets[k * per:(k + 1) * per]
        tiles = []
        for p in chunk:
            im = Image.open(p).convert("RGB")
            tiles.append(im.resize((GRID_W, int(im.height * GRID_W / im.width))))
        th = max(t.height for t in tiles)
        rows = math.ceil(len(tiles) / GRID_COLS)
        page = Image.new("RGB", (GRID_COLS * GRID_W, rows * th), (40, 40, 40))
        for i, t in enumerate(tiles):
            page.paste(t, ((i % GRID_COLS) * GRID_W, (i // GRID_COLS) * th))
        out = CONTACT_DIR / f"_grid_{k}.png"
        page.save(out)
        outs.append(out)
    return outs


SEED_RETRIES = int(os.environ.get("RL_GT_SEED_RETRIES", "2"))
# a frame darker than this (mean luma) is re-rolled with another seed: some seeds
# pick near-black wall tiles (115/345/348) and the fixtures vanish.
LUMA_DARK = float(os.environ.get("RL_GT_LUMA_DARK", "40"))
# ... or when most pixels are near-black even if a bright window lifts the mean
# (115 in v2b: mean 63 yet fixtures invisible)
DARK_FRAC = float(os.environ.get("RL_GT_DARK_FRAC", "0.55"))
DARK_PX = 30


def _darkness(png: Path) -> tuple[float, float]:
    """(mean luma, fraction of pixels below DARK_PX)."""
    im = Image.open(png).convert("L")
    hist = im.histogram()
    n = sum(hist)
    return ImageStat.Stat(im).mean[0], (sum(hist[:DARK_PX]) / n if n else 0.0)


def _too_dark(png: Path) -> bool:
    mean, frac = _darkness(png)
    return mean < LUMA_DARK or frac > DARK_FRAC


def _render_one(stem: str, slot: int) -> tuple[str, str]:
    cfg = IndoorConfig.from_json((GT_DIR / f"{stem}.json").read_text())
    # Infinigen's solver has a rare stock crash (Addition name collision:
    # `assert target_name not in state.objs`, seen on 313). It is seed-dependent,
    # so retry with a bumped seed instead of losing the scene; the retry seed is
    # recorded in <stem>/seed_used.txt so the report can flag it.
    last: Exception | None = None
    for attempt in range(SEED_RETRIES + 1):
        try:
            png = render(cfg, OUT_DIR / stem, stub=False, slot=slot)
        except RuntimeError as e:  # a stage failed; try another seed
            last = e
            print(f"[retry] {stem}: attempt {attempt} crashed; bumping seed", flush=True)
            cfg.seed = (cfg.seed + 1000) % (2 ** 32)
            continue
        luma, dfrac = _darkness(png)
        (OUT_DIR / stem / "seed_used.txt").write_text(
            f"{cfg.seed} (attempt {attempt}, luma {luma:.1f}, dark_frac {dfrac:.2f})\n")
        _contact(stem, png)
        if _too_dark(png) and attempt < SEED_RETRIES:
            print(f"[retry] {stem}: attempt {attempt} too dark (luma {luma:.1f}, dark {dfrac:.2f}); bumping seed", flush=True)
            cfg.seed = (cfg.seed + 1000) % (2 ** 32)
            continue
        return stem, "ok"
    if last is not None and not _render_png(stem):
        raise RuntimeError(f"{stem}: all {SEED_RETRIES + 1} seeds failed: {last}")
    return stem, "ok"  # dark but rendered; the report flags the luma


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=int(os.environ.get("RL_RENDER_WORKERS", "8")))
    ap.add_argument("--only", nargs="*", default=None, help="subset of stems to render")
    ap.add_argument("--report-only", action="store_true", help="skip rendering; re-analyze")
    args = ap.parse_args()

    all_stems = sorted(p.stem for p in GT_DIR.glob("*.json"))
    stems = args.only or all_stems
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ok, err = 0, 0
    if not args.report_only:
        # a bounded set of GPU slots; each in-flight render owns one.
        slots: queue.Queue[int] = queue.Queue()
        for s in range(args.workers):
            slots.put(s)

        def task(stem: str) -> tuple[str, str]:
            slot = slots.get()
            try:
                return _render_one(stem, slot)
            except Exception as e:  # noqa: BLE001 - keep the batch going, report per-scene
                return stem, f"ERROR: {e}\n{traceback.format_exc(limit=2)}"
            finally:
                slots.put(slot)

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(task, s) for s in stems]
            for f in as_completed(futs):
                stem, status = f.result()
                if status == "ok":
                    ok += 1
                    print(f"[ok]   {stem}", flush=True)
                else:
                    err += 1
                    print(f"[FAIL] {stem}: {status}", flush=True)

    # report over EVERYTHING on disk (so partial re-renders keep the full picture)
    report = {}
    if REPORT.exists():
        try:
            report = json.loads(REPORT.read_text())
        except json.JSONDecodeError:
            report = {}
    for stem in stems:
        try:
            report[stem] = _analyze(stem)
        except Exception as e:  # noqa: BLE001
            report[stem] = {"stem": stem, "status": "FAIL", "fails": [f"analyze error: {e}"]}
    REPORT.write_text(json.dumps(report, indent=1, sort_keys=True))
    pages = _grid(all_stems)

    n_fail = sum(1 for s in stems if report[s]["status"] == "FAIL")
    print("\n--- automatic report ---")
    for s in stems:
        r = report[s]
        extra = f" area={r.get('room_area')} luma={r.get('luma')} clear={r.get('cam_clearance')}"
        flags = "; ".join(r.get("fails", []) + [f"warn: {w}" for w in r.get("warns", [])])
        print(f"[{r['status']:4s}] {s:45s}{extra}  {flags}")
    if not args.report_only:
        print(f"\nrendered {ok}/{len(stems)} ({err} failed).")
    print(f"auto-FAIL: {n_fail}/{len(stems)}. Contact sheets: {CONTACT_DIR}; "
          f"montage pages: {len(pages)}; report: {REPORT}")
    return 1 if (err or n_fail) else 0


if __name__ == "__main__":
    raise SystemExit(main())
