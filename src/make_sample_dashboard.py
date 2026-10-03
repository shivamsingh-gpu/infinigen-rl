#!/usr/bin/env python
"""Build a fresh TensorBoard dashboard: each rendered sample as an IMAGES panel
(render + prompt burned in) plus a TEXT entry, keyed by room/ref.

Usage:
  python make_sample_dashboard.py --samples <sample.json> --renders <sample_renders_dir> \
         --val <val.parquet> --logdir <out_tb_dir>
"""
import argparse, json, textwrap, numpy as np, shutil
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from torch.utils.tensorboard import SummaryWriter

def F(sz, b=False):
    p = "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if b else "")
    try: return ImageFont.truetype(p, sz)
    except Exception: return ImageFont.load_default()

def render_png(renders, room, ref):
    pngs = sorted((Path(renders)/room/ref/"frames"/"Image"/"camera_0").glob("Image_*.png"))
    return pngs[0] if pngs else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True); ap.add_argument("--renders", required=True)
    ap.add_argument("--val", required=True); ap.add_argument("--logdir", required=True)
    a = ap.parse_args()
    scores = {r["ref"]: r for r in json.load(open(a.samples))}
    df = pd.read_parquet(a.val); prompts = {}
    for r in df.itertuples(index=False):
        u = [m["content"] for m in r.prompt if m["role"] == "user"][0]
        prompts[r.extra_info["ref_stem"]] = u.split("\n",1)[-1].strip() if "\n" in u else u
    out = Path(a.logdir)
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    fb, fl, fm = F(22), F(19, True), F(28, True)
    w = SummaryWriter(log_dir=str(out)); n = 0
    for rd in sorted(Path(a.renders).iterdir()):
        if not rd.is_dir() or rd.name == "contact": continue
        for ref_dir in sorted(rd.iterdir()):
            p = render_png(a.renders, rd.name, ref_dir.name)
            if not p: continue
            room, ref = rd.name, ref_dir.name
            prompt = prompts.get(ref, "(prompt not found)"); sc = scores.get(ref, {})
            top = Image.open(p).convert("RGB"); H = 640
            top = top.resize((int(top.width*H/top.height), H)); W = top.width
            lines = textwrap.wrap(prompt, width=max(60, W//12)); pad, lh = 20, 28
            card = Image.new("RGB", (W, H + pad*3 + lh*(len(lines)+2)), (24,24,28))
            card.paste(top, (0,0)); d = ImageDraw.Draw(card); y = H+pad
            d.text((pad,y), f"{room} / {ref}", font=fm, fill=(240,240,160)); y += lh+6
            d.text((pad,y), f"score={sc.get('score')} set_match={sc.get('set_match')} parse_ok={sc.get('parse_ok')}", font=fb, fill=(150,210,150)); y += lh+6
            for ln in lines: d.text((pad,y), ln, font=fb, fill=(220,220,220)); y += lh
            tag = f"{room}/{ref}"
            w.add_image(tag, np.asarray(card, np.uint8), 0, dataformats="HWC")
            w.add_text(tag, f"**{room} / {ref}** - score {sc.get('score')}\n\n{prompt}", 0)
            n += 1
    w.close(); print(f"wrote {n} panels to {out}")

if __name__ == "__main__":
    main()
