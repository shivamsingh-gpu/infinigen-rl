#!/usr/bin/env bash
# GRPO | Qwen3.5-2B | FSDP | Infinigen bathroom layout (text -> IndoorConfig JSON)
# Adapted from demo_qwen35_gsm8k/run_qwen3_5_2b_gsm8k_fsdp.sh:
#   - swaps the GSM8K dataset for the bathroom parquet (build_bathroom_dataset.py)
#   - swaps the built-in gsm8k rule reward for our custom schema-vs-GT reward
#     (reward_bathroom.compute_score), which returns a dict so every sub-metric
#     (presence/placement/set_match/forbidden_penalty/...) streams to TensorBoard
#   - shorter response length (schemas are small JSON)
set -xeuo pipefail

########################### environment fixes (identical to the GSM8K demo) ###########################
COMPAT_DIR=${COMPAT_DIR:-/nara-efs/marketing/shhsing/verl/demo_qwen35_gsm8k/cuda_compat/usr/local/cuda-13.0/compat}
if [ -d "${COMPAT_DIR}" ]; then export LD_LIBRARY_PATH="${COMPAT_DIR}:${LD_LIBRARY_PATH:-}"; fi
VENV_BIN=${VENV_BIN:-/nara-efs/marketing/shhsing/verl/.venv/bin}
export PATH="${VENV_BIN}:${PATH}"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
unset ROCR_VISIBLE_DEVICES || true
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
########################### end environment fixes ###########################

########################### user-adjustable ###########################
DEVICE=${DEVICE:-gpu}
INFER_BACKEND=${INFER_BACKEND:-vllm}
PROJECT_NAME=${PROJECT_NAME:-GRPO-Infinigen-bathroom}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-Qwen3.5-2B-GRPO-bathroom}

NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-2}
NNODES=${NNODES:-1}
GEN_TP=${GEN_TP:-1}
SP_SIZE=${SP_SIZE:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.5}

BATH_HOME=${BATH_HOME:-/nara-efs/marketing/shhsing/verl/rl_infinigen_bathroom}
# reuse the already-downloaded, proven Qwen3.5-2B checkpoint from the GSM8K demo
MODEL_PATH=${MODEL_PATH:-/nara-efs/marketing/shhsing/verl/demo_qwen35_gsm8k/models/Qwen3.5-2B}
CKPTS_DIR=${CKPTS_DIR:-"${BATH_HOME}/ckpts/${EXPERIMENT_NAME}"}
LOG_DIR=${LOG_DIR:-"${BATH_HOME}/logs"}
TRAIN_FILE=${TRAIN_FILE:-"${BATH_HOME}/data/train.parquet"}
TEST_FILE=${TEST_FILE:-"${BATH_HOME}/data/val.parquet"}
# custom reward module (lives with the Infinigen scaffold). reward_indoor is the
# generalized reward: it reads room_type from the GT and dispatches to the
# per-room ontology (indoor_ontology), so it scores bathroom AND bedroom (and
# future room types) identically to the old reward_bathroom for bathroom data.
REWARD_PATH=${REWARD_PATH:-/nara-efs/marketing/shhsing/infinigen/rl_infinigen_beginner/scripts/reward_indoor.py}
########################### end user-adjustable ###########################

n_devices_per_node=${NDEVICES_PER_NODE}
fsdp_size=${NDEVICES_PER_NODE}
start_time=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p "${LOG_DIR}"

# TensorBoard: every logged metric (score + all reward sub-metrics, losses,
# grad_norm, entropy, kl, response lengths, throughput, lr) is written as a
# scalar to this dir. One timestamped subdir per run.
TB_DIR=${TB_DIR:-"${BATH_HOME}/tb/${EXPERIMENT_NAME}/${start_time}"}
export TENSORBOARD_DIR="${TB_DIR}"
mkdir -p "${TENSORBOARD_DIR}"

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=16
    data.max_prompt_length=2048
    data.max_response_length=640
    data.filter_overlong_prompts=True
    data.filter_overlong_prompts_workers=8
    data.truncation='error'
    data.shuffle=True
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=8
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.kl_loss_coef=0.02
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size}
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.fsdp_config.offload_policy=True
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

REF=(
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.fsdp_config.offload_policy=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.prompt_length=2048
    actor_rollout_ref.rollout.response_length=640
    actor_rollout_ref.rollout.ignore_eos=False
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=8
    actor_rollout_ref.rollout.max_num_batched_tokens=4096
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=6144
)

# custom schema-vs-ground-truth reward (returns a dict -> sub-metrics logged).
# reward.py reads config.reward.custom_reward_function, so keys are prefixed `reward.`
REWARD=(
    reward.custom_reward_function.path="${REWARD_PATH}"
    reward.custom_reward_function.name=compute_score
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger=['console','tensorboard']
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node=${n_devices_per_node}
    trainer.nnodes=${NNODES}
    trainer.balance_batch=False
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.val_before_train=False
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.total_epochs=${TOTAL_EPOCHS:-30}
    trainer.total_training_steps=${TOTAL_STEPS:-60}
)

VENV_PY=${VENV_PY:-/nara-efs/marketing/shhsing/verl/.venv/bin/python}
LAUNCH=("${VENV_PY}")
RAY=(ray_kwargs.ray_init.runtime_env.py_executable=null)

"${LAUNCH[@]}" -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${ROLLOUT[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${RAY[@]}" \
    "$@" 2>&1 | tee "${LOG_DIR}/bathroom-grpo-${start_time}.log"
