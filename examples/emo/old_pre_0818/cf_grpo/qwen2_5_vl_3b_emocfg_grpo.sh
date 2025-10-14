#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

# MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path
MODEL_PATH=/workspace/saves/qwen2_5vl-3b/full_sft/1epoch/checkpoint-16

python3 -m verl.trainer.main \
    config=examples/config_emo_cfg.yaml \
    data.train_files=fuyyy74/EmoSet3steps2k@train \
    data.val_files=fuyyy74/EmoSet3steps2k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    trainer.experiment_name=qwen2_5_vl_3b_0e_20e_from_sft_1epoch_0730_grpo \
    trainer.n_gpus_per_node=4


