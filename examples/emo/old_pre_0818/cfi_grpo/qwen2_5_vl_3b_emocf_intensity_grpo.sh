#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

# MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path
MODEL_PATH=/workspace/easyr1_self/checkpoints/easy_r1/qwen2_5_vl_3b_emocf_intensity_grpo/global_step_60/actor/huggingface

python3 -m verl.trainer.main \
    config=examples/config_emo_intensity.yaml \
    data.train_files=fuyyy74/EmoSet3steps2k@train \
    data.val_files=fuyyy74/EmoSet3steps2k@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    trainer.experiment_name=qwen2_5_vl_3b_emocf_intensity_2stage_grpo \
    trainer.n_gpus_per_node=4     # TODO: change it to your number of GPUs per node \

