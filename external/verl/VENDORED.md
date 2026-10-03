# Vendored dependency: verl

This is an **unmodified** copy of the verl training framework, vendored so the GRPO
trainer (`verl.trainer.main_ppo`) ships with this repo and runs offline.

- Upstream: https://github.com/verl-project/verl.git
- Pinned commit: `3efe38c759c14622fd1b2c9e3679f2d02f86bdac`
- git describe: `v0.8.0-442-g3efe38c7`
- Local modifications at vendor time: **none** (clean checkout).
- License: Apache-2.0 (see ./LICENSE, ./Notice.txt).

**Do not edit files here.** Our integration touches verl only through:
  - launch/run_grpo_fsdp.sh (hydra overrides + `reward.custom_reward_function.path`),
  - src/reward_indoor.py (the custom reward, passed by path).
To update verl: re-copy from upstream at a new pinned commit and update this file.

Third-party runtime deps (torch, ray, vllm, flash-attn, …) are NOT vendored; they come
from $VENV (see config/paths.env). `python -m verl.trainer.main_ppo` imports the verl
*source* from this directory (verl is run from source, not pip-installed).
