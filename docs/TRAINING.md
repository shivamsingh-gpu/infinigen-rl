# Training

GRPO on Qwen3.5-2B via verl (FSDP2 + vLLM rollout). Driver: `launch/run_grpo_fsdp.sh`
(it defaults the reward to the repo's `src/reward_indoor.py` and works for all rooms).
Durable wrapper: `launch/train.sbatch` (forwards `"$@"` so you can append hydra overrides).

**Where the actual training algorithm is:** `launch/run_grpo_fsdp.sh` sets all the knobs and then
calls `python -m verl.trainer.main_ppo` — the GRPO loop itself. That module is **vendored** in this
repo at `external/verl/verl/trainer/main_ppo.py` (pinned, unmodified; see `external/verl/VENDORED.md`).
`config/paths.env` puts `$VERL_SRC` (= `external/verl`) first on `PYTHONPATH`, so the in-repo copy is
used — not any external verl checkout. Third-party deps (torch/ray/vllm) still come from `$VENV`.
Outputs (ckpts/logs/tb) default to `$REPO_ROOT/runs/` (override via `BATH_HOME`/`CKPTS_DIR`/`TB_DIR`).

## What one step is
- `train_batch_size=16` prompts/step, `rollout.n=8` samples/prompt → **128 generations/step**.
- Score all 128 with `reward_indoor`; advantage = reward − prompt-group mean (GRPO, no critic).
- Update over `ppo_mini_batch_size=8` (2 gradient steps/step) with KL to a frozen reference.
- 166 train rows ÷ 16 ≈ 10.4 steps/epoch ⇒ 60 steps ≈ 5.8 epochs. `total_training_steps` is the
  binding limit (vs `total_epochs=30`).

## Key defaults (all env-overridable; see top of run_grpo_fsdp.sh)
- Model `demo_qwen35_gsm8k/models/Qwen3.5-2B`; GRPO; FSDP2; vLLM `n=8`.
- `lr=1e-6`, `kl_loss_coef=0.02` (`low_var_kl`), `entropy_coeff=0`.
- `max_prompt_length=2048`, `max_response_length=640`.
- CUDA-compat `LD_LIBRARY_PATH`, `HF_HUB_DISABLE_XET=1` baked in.
- Logs: TensorBoard `tb/<EXPERIMENT_NAME>/<ts>/`; text log in `logs/`.

## Launch — merged 5-room policy (recommended), 8-GPU p4 node
```bash
source config/paths.env
cd $VERL_ROOT
sbatch --job-name=qwen35-grpo-indoor \
  --partition=p4-80-main --qos=batch --nodes=1 --gres=gpu:8 --cpus-per-task=96 --time=08:00:00 \
  --export=ALL,NDEVICES_PER_NODE=8,\
TRAIN_FILE=$DATA_ROOT/train.parquet,\
TEST_FILE=$DATA_ROOT/val.parquet,\
EXPERIMENT_NAME=Qwen3.5-2B-GRPO-indoor,\
PROJECT_NAME=GRPO-Infinigen-indoor,\
CKPTS_DIR=$CKPT_ROOT/Qwen3.5-2B-GRPO-indoor \
  $REPO_ROOT/launch/train.sbatch trainer.save_freq=20 trainer.test_freq=20
```
- `NDEVICES_PER_NODE=8` → `fsdp_size=8` and `n_gpus_per_node=8` (default in the script is 2).
- `save_freq=20` / `test_freq=20`: checkpoint + validate on the 25-row val set every 20 steps.
- The partition `p4-80-main`/qos `batch` is non-preempt. (`p5-141-nara-preempt`/`team-nara-preempt`
  works too but can be preempted — then rely on checkpoints to resume by resubmitting.)

### Per-room run
Point `TRAIN_FILE`/`TEST_FILE` at that room's parquet and rename `EXPERIMENT_NAME`.

## Monitoring
```bash
tail -f $VERL_ROOT/rl_infinigen_bathroom/logs/slurm-<jobid>.out      # console
./launch/tensorboard.sh                                             # curves on :6007
```
Health signals (reference run): `critic/rewards/mean` 0.63→0.96; `actor/grad_norm` ~1.7 stable;
`actor/kl_loss` 0.0001→0.035; `actor/entropy` 0.17→0.04; `rollout_corr/*` ≈1.0 (vLLM↔trainer agree).
If `grad_norm` AND `kl_loss` rise rapidly together → instability/reward-hacking: lower `lr`, raise
`kl_loss_coef`, and read a few generations. See the curve glossary in `INFERENCE.md`.

## Checkpoints
Land in `$CKPT_ROOT/<EXPERIMENT_NAME>/global_step_<N>/` (sharded FSDP: `actor/`, `data.pt`,
`latest_checkpointed_iteration.txt`). Merge to HF before inference — see `INFERENCE.md`.

## Harmless log noise
`multiprocess/util.py ... Device or resource busy` (pymp tempdir cleanup on NFS) and optional-engine
`ModuleNotFoundError` (torchtitan/megatron/veomni — unused; this run is FSDP2). Not failures.
