set -x

MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"  # replace it with your local file path

export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
export NCCL_P2P_DISABLE=1
export MAX_JOBS=16
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export DEBIAN_FRONTEND=noninteractive
export NODE_OPTIONS=""
export PIP_ROOT_USER_ACTION=ignore
export HF_HUB_ENABLE_HF_TRANSFER="1"


python -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=hiyouga/geometry3k@train \
    data.val_files=hiyouga/geometry3k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    trainer.experiment_name=qwen2_5_vl_3b_geo_grpo \
    trainer.n_gpus_per_node=8
