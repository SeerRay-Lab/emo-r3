#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

ROLLOUT_NUMBER=8
TRAING_EPOCH=20
EXPERIMENT_NAME=qwen2_5_vl_3b_emoset_${ROLLOUT_NUMBER}
MODEL_PATH=Qwen2.5-VL-3B-Instruct
CHECKPONT_PATH=${EXPERIMENT_NAME}

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=fuyyy74/EmoSet2k@train \
    data.val_files=fuyyy74/EmoSet2k@test \
    data.format_prompt=./examples/format_prompt/emor3.jinja \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.is_rethink=true \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.n=${ROLLOUT_NUMBER} \
    worker.reward.reward_function=./examples/reward_function/emor3.py:compute_score \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=${TRAING_EPOCH} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.save_checkpoint_path=${CHECKPONT_PATH}