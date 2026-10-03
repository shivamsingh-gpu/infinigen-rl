#!/usr/bin/env python
"""Sample layouts from a trained indoor-GRPO checkpoint and score them with reward_indoor.

Loads a merged HF model, replays N val prompts, generates one layout each (low-temp),
scores vs GT with the same reward used in training, and prints a per-room summary.
"""
import argparse, json, sys, os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reward_indoor as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max_new", type=int, default=640)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    if args.n and args.n < len(df):
        # even spread across rooms
        df = df.groupby(df["extra_info"].map(lambda e: e["room_type"]), group_keys=False)\
               .apply(lambda g: g.head(max(1, args.n // df["extra_info"].map(lambda e: e["room_type"]).nunique())))
    print(f"sampling {len(df)} prompts from {args.parquet}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    rows, by_room = [], {}
    for _, r in df.iterrows():
        msgs = [{"role": m["role"], "content": m["content"]} for m in r["prompt"]]
        gt = r["reward_model"]["ground_truth"]
        room = r["extra_info"]["room_type"]
        ref = r["extra_info"]["ref_stem"]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to("cuda")
        in_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new,
                                 do_sample=args.temperature > 0, temperature=max(args.temperature, 1e-5),
                                 top_p=0.95, pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0][in_len:], skip_special_tokens=True)
        sc = R.compute_score("infinigen_indoor", gen, gt, {})
        score = sc["score"] if isinstance(sc, dict) else sc
        gt_sc = R.compute_score("infinigen_indoor", gt, gt, {})
        gt_score = gt_sc["score"] if isinstance(gt_sc, dict) else gt_sc
        by_room.setdefault(room, []).append(score)
        rows.append({"room": room, "ref": ref, "score": round(score, 3),
                     "gt_ceiling": round(gt_score, 3),
                     "parse_ok": sc.get("parse_ok"), "presence": round(sc.get("presence", 0), 2),
                     "placement": round(sc.get("placement", 0), 2), "set_match": round(sc.get("set_match", 0), 2),
                     "n_forbidden": sc.get("n_forbidden"), "gen": gen.strip()})
        print(f"[{room:11s}] {ref:40s} score={score:.3f} (ceil {gt_score:.2f}) "
              f"parse={sc.get('parse_ok')} pres={sc.get('presence'):.2f} "
              f"place={sc.get('placement'):.2f} set={sc.get('set_match'):.2f} forb={sc.get('n_forbidden')}", flush=True)

    print("\n=== per-room mean score ===")
    for room, xs in sorted(by_room.items()):
        print(f"  {room:12s} mean={sum(xs)/len(xs):.3f}  n={len(xs)}")
    allxs = [x for xs in by_room.values() for x in xs]
    print(f"  {'OVERALL':12s} mean={sum(allxs)/len(allxs):.3f}  n={len(allxs)}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(rows, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
