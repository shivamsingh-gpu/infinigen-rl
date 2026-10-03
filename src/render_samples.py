#!/usr/bin/env python
"""Render policy-generated layouts (from sample_indoor_policy.py --out) with Infinigen
and build a reference | render contact sheet per room, plus montage grids.

Uses the same deterministic corner camera + exclusive population as GT validation,
so what you see is exactly what the generated schema encodes.

Run on a GPU node:
  RL_INDOOR_EXCLUSIVE=1 python render_samples.py --samples <sample.json> --workers 8
"""
from __future__ import annotations
import argparse, json, os, queue, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image, ImageDraw

from indoor_config_space import IndoorConfig
from render_indoor import render

BASE = Path(__file__).resolve().parents[1]
_REF_SUFFIX = {"Bathroom": "bath", "Bedroom": "bedroom", "LivingRoom": "living",
               "Kitchen": "kitchen", "DiningRoom": "dining"}
SEED_RETRIES = 2


def ref_path(room_type: str, stem: str) -> Path:
    return BASE / "references" / f"ext_indoor_{_REF_SUFFIX.get(room_type, room_type.lower())}" / f"{stem}.png"


def contact(ref: Path, render_png: Path, out: Path, label: str):
    H = 512
    def load(p, fallback):
        if p and p.exists():
            im = Image.open(p).convert("RGB")
            return im.resize((int(im.width * H / im.height), H))
        im = Image.new("RGB", (H, H), fallback); ImageDraw.Draw(im).text((10, 10), "(missing)", fill=(240, 240, 240)); return im
    a, b = load(ref, (60, 60, 60)), load(render_png, (90, 20, 20))
    card = Image.new("RGB", (a.width + b.width + 12, H + 26), (18, 18, 18))
    card.paste(a, (0, 26)); card.paste(b, (a.width + 12, 26))
    ImageDraw.Draw(card).text((6, 6), f"{label}   [ reference | render ]", fill=(230, 230, 230))
    out.parent.mkdir(parents=True, exist_ok=True); card.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out_dir", default=str(BASE / "sample_renders"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    rows = json.load(open(args.samples))
    if args.only:
        rows = [r for r in rows if r["ref"] in args.only]
    out_root = Path(args.out_dir)
    print(f"rendering {len(rows)} generated layouts -> {out_root}", flush=True)

    slots = queue.Queue()
    for s in range(args.workers):
        slots.put(s)

    def work(r):
        slot = slots.get()
        stem, room = r["ref"], r["room"]
        od = out_root / room / stem
        try:
            cfg = IndoorConfig.from_json(r["gen"])
            png, last = None, None
            for attempt in range(SEED_RETRIES + 1):
                try:
                    png = render(cfg, od, stub=False, slot=slot)
                    break
                except RuntimeError as e:
                    last = e; cfg.seed = (cfg.seed + 1000) % (2 ** 32)
                    print(f"[retry] {stem}: attempt {attempt} crashed; bumping seed", flush=True)
            if png is None:
                raise RuntimeError(f"{stem}: all seeds failed: {last}")
            contact(ref_path(room, stem), png, out_root / "contact" / f"{room}__{stem}.png",
                    f"{room} / {stem}  score={r.get('score')}")
            return stem, "OK"
        except Exception as e:
            traceback.print_exc()
            return stem, f"FAIL: {e}"
        finally:
            slots.put(slot)

    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, r): r["ref"] for r in rows}
        for f in as_completed(futs):
            stem, status = f.result()
            results[stem] = status
            print(f"[{status.split(':')[0]:4s}] {stem}", flush=True)

    # montage all contact sheets
    sheets = sorted((out_root / "contact").glob("*.png"))
    if sheets:
        ims = [Image.open(p).convert("RGB") for p in sheets]
        w = max(i.width for i in ims); rowh = ims[0].height
        grid = Image.new("RGB", (w, rowh * len(ims) + 8 * len(ims)), (10, 10, 10))
        y = 0
        for im in ims:
            grid.paste(im, (0, y)); y += im.height + 8
        grid.save(out_root / "contact" / "_grid_all.png")
        print(f"grid -> {out_root/'contact'/'_grid_all.png'}", flush=True)

    ok = sum(1 for v in results.values() if v == "OK")
    print(f"\n=== {ok}/{len(results)} rendered OK ===")
    for k, v in sorted(results.items()):
        if v != "OK":
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
