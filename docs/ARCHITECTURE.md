# Architecture

## Goal
Train an LLM to map a natural-language room description → a structured 3D layout schema that
Infinigen's constraint solver can turn into a rendered room.

## Problem
- 3D interior layout is slow, manual, expert-driven.
- Off-the-shelf, an LLM emits invalid JSON, wrong fixtures, or physically impossible placements.
- There is no large labeled "description → layout" corpus, and "good layout" is not a simple loss.
- **Solution: RL.** Reward good layouts with a deterministic rule instead of hand-labeling them.

## The three phases (and the key design decision)

### 1. Data (offline, one-time)
Reference room photos (5 types: bathroom, bedroom, living, kitchen, dining) are turned into
**GT `IndoorConfig` schemas** by `author_gt_schemas.py`, then packed into prompt→GT parquet rows
by `build_indoor_dataset.py`. See `DATA.md`.

### 2. Training loop (verl GRPO — text only)
Per step: sample 16 prompts → policy generates 8 layouts each (vLLM) → `reward_indoor` scores all
128 against their GT → GRPO computes group-relative advantages → FSDP2 updates the policy with a
KL anchor to a frozen reference. See `TRAINING.md` and `REWARD.md`.

**Key design decision — rendering is NOT in the loop.** The reward is *schema-vs-schema* (compare
the generated JSON to the GT JSON), which is:
- **fast** (pure Python, no Blender) — essential at 128 rollouts/step,
- **deterministic** — stable, reproducible advantages,
- **requires a GT schema per prompt** — which is why the data phase exists.

The trade-off: the reward optimizes *layout composition* (right fixtures, right surfaces, no junk),
not *physical/visual quality*. Geometric validity is handled separately by the render/eyeball gate
in phase 3, not by the reward. (Moving to a render- or VLM-based reward is the natural next lever;
the blocker is render throughput — see the roadmap note at the bottom.)

### 3. Eval / inference (offline)
Merge the FSDP checkpoint → HF, sample held-out val prompts, render the generated layouts in
Infinigen/Blender, and build a TensorBoard image+prompt gallery. See `INFERENCE.md`.

## Component map

| Component | File | Role |
|---|---|---|
| Schema | `src/indoor_config_space.py` | `IndoorConfig` dataclass; `SCHEMA_FOR_LLM` = the system-prompt spec the policy must follow |
| Ontology | `src/indoor_ontology.py` | `ONTOLOGY[room_type]`: which factories are core/optional/forbidden, surface classes, object budget |
| Reward | `src/reward_indoor.py` | `compute_score()` — the GRPO reward; dispatches on room_type |
| Dataset | `src/build_indoor_dataset.py` | GT schemas (+captions) → train/val parquet |
| Policy | Qwen3.5-2B (`$BASE_MODEL`) | text-in → JSON-out LLM; trained by verl GRPO |
| Trainer | verl (`$VERL_ROOT`) | GRPO/FSDP2/vLLM; launched via `launch/run_grpo_fsdp.sh` |
| Renderer | infinigen (`$INFINIGEN_ROOT`) + `src/render_indoor.py` | compiles a schema → scene → image |
| RL hook | `patches/infinigen/rl_inject.py` | compiles the JSON schema into Infinigen's constraint DSL (env-gated) |

## Data flow of one sample (what sees what)
- **Policy input:** the two chat messages only — `system` (schema spec) + `user` (room description).
- **Policy output:** one JSON `IndoorConfig` (up to 640 tokens).
- **The GT is never shown to the policy.** It lives in `reward_model.ground_truth` and is read only
  by `reward_indoor` *after* generation. The learning signal reaches the model only via the reward
  → GRPO gradient. This is why the held-out val score is a real generalization number.
- **No images anywhere in training.** Reference photos only shaped the GT captions offline.

## Room-type-agnostic by construction
`indoor_ontology.ONTOLOGY[room_type]` + `reward_indoor` dispatch + per-room sizing in
`render_indoor.ROOM_SIZING` mean one merged parquet trains a single policy over all 5 room types
with the same reward. Add a room type by extending the ontology + authoring GT schemas; nothing
else changes.

## Roadmap note (where this is heading)
- **Image-conditioned policy:** swap Qwen for a VLM, feed the reference image, reuse the existing
  (image, GT) pairs; reward can stay schema-vs-schema at first.
- **Image/VLM reward:** add a render-based term as a post-saturation fine-tune (val saturates by
  ~step 20, so the dense schema reward does the early climb cheaply). Gated on fast-render throughput.
- **Open-vocab / personalized assets:** retrieval (Objaverse/3D-FUTURE) then image-to-3D (Hunyuan3D)
  wrapped as custom `AssetFactory`s; `set_match` moves from exact-string to semantic matching.
- **Physics:** add a cheap in-loop physical-plausibility reward term (interpenetration + support +
  satisfiability) to attack solver `viol` failures and floating objects; cloth-sim bake for eval renders.
