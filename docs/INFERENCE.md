# Inference / Eval

Four steps: **merge → sample → render → dashboard.**

## 1. Merge the FSDP checkpoint → HF model
```bash
source config/paths.env
CKPT=$CKPT_ROOT/Qwen3.5-2B-GRPO-indoor/global_step_60/actor
OUT=$CKPT_ROOT/Qwen3.5-2B-GRPO-indoor/merged_hf_step60
$PY -m verl.model_merger merge --backend fsdp --local_dir "$CKPT" --target_dir "$OUT"
```
Produces a standard HF model (`model.safetensors`, tokenizer, config). **Needs the CUDA-compat
env** (`config/paths.env` + the `LD_LIBRARY_PATH` from `run_grpo_fsdp.sh`), else "NVIDIA driver too old".

## 2. Sample layouts and score them
`src/sample_indoor_policy.py` loads the merged model, replays val prompts, generates one layout
each (low temp), and scores with `reward_indoor`.
```bash
$PY src/sample_indoor_policy.py \
  --model $CKPT_ROOT/Qwen3.5-2B-GRPO-indoor/merged_hf_step60 \
  --parquet $DATA_ROOT/val.parquet \
  --n 15 --temperature 0.2 \
  --out $CKPT_ROOT/Qwen3.5-2B-GRPO-indoor/sample_step60.json
```
Prints per-sample + per-room mean score and writes a JSON with each `{room, ref, score, gen}`.
(Needs a GPU — submit via `launch/sample.sbatch` on nodes where interactive `srun` is blocked.)
Reference run: held-out val mean **0.985**.

> Gotcha: use `apply_chat_template(..., return_dict=True)` + `generate(**enc)`; passing the bare
> tensor raises `KeyError: 'shape'` on this transformers version. Already handled in the script.

## 3. Render the generated layouts (Infinigen + Blender)
`src/render_samples.py` builds an `IndoorConfig` from each generated layout and renders it with the
deterministic corner camera + exclusive population, into reference|render contact sheets.
```bash
cd $SCAFFOLD && sbatch $REPO_ROOT/launch/render_samples.sbatch   # 8-GPU node
# output: sample_renders/<Room>/<ref>/frames/Image/camera_0/Image_*.png
#         sample_renders/contact/<Room>__<ref>.png  + _grid_all.png
```
Needs the **infinigen driver venv** (`$INFINIGEN_VENV`) and the applied infinigen patches
(`INFINIGEN_PATCHES.md`). Complex multi-fixture scenes can hold `viol=1.0` in the solver and run
long — cap/skip them.

## 4. TensorBoard sample gallery (render + prompt)
```bash
$PY src/make_sample_dashboard.py \
  --samples $CKPT_ROOT/Qwen3.5-2B-GRPO-indoor/sample_step60.json \
  --renders $SCAFFOLD/sample_renders \
  --val $DATA_ROOT/val.parquet \
  --logdir $VERL_ROOT/rl_infinigen_bathroom/tb_samples/Qwen3.5-2B-GRPO-indoor
./launch/tensorboard.sh      # curves :6007, gallery :6008  (localhost; tunnel in)
```
IMAGES tab = each render with its prompt burned in; TEXT tab = full prompts, keyed by `room/ref`.

Tunnel from your laptop:
```bash
ssh -N -L 6007:127.0.0.1:6007 -L 6008:127.0.0.1:6008 <you>@<node-host>
```
> Login nodes here are resource-starved; the fast TB loader panics (Rayon). `tensorboard.sh`
> already forces `--load_fast=false`.

## Single-prompt inference (no GT needed)
To generate a layout for an arbitrary new description, build a 1-row prompt with the same
`system` spec (`indoor_config_space.SCHEMA_FOR_LLM`) + your `user` text, run it through the merged
model (step 2 without `--parquet` scoring), then render it (step 3). The reward/scoring is only
needed when you have a GT to compare against.

## Training-curve glossary (TensorBoard :6007)
- `critic/rewards/mean` — headline reward (↑ good). `max/min` show per-batch spread GRPO needs.
- `actor/grad_norm` — update magnitude; bounded = stable.
- `actor/kl_loss`, `kl_coef` — divergence from frozen ref + its weight.
- `actor/entropy` — output randomness; gently ↓ as it converges (crash → mode collapse).
- `actor/pg_loss`, `pg_clipfrac` — policy-gradient loss / fraction hitting PPO clip.
- `response_length/*` — generated-token length (and clip_ratio at the 640 cap).
- `rollout_corr/*`, `training/rollout_*` — vLLM↔trainer agreement (≈1.0 = healthy).
- `perf/*`, `timing_s/*`, `global_seqlen/*` — speed/throughput/load-balance telemetry.
- `val-core/.../reward/mean@1` — held-out reward (1 sample/prompt) at `test_freq` steps.
- `val-aux/.../*` — the per-term breakdown of the val reward (penalties flat at 0 = good).
