set -x
MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path
export NCCL_P2P_DISABLE=1
export MAX_JOBS=32
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
    trainer.experiment_name=qwen2_5_vl_7b_geo_grpo \
    trainer.n_gpus_per_node=8
