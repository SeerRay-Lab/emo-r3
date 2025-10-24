#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

export WANDB_API_KEY=0aedaabc5db114ea5922799978efb09dc1d52b63
export WANDB_DIR=/vlm-ssd/fangyiyang/easyr1_emo/wandb_log
export WANDB_ENTITY=fuyyy-wuhan-university

export RAY_TMPDIR=/vlm-ssd/fangyiyang/tmp_ray
mkdir -p $RAY_TMPDIR

ROLLOUT_NUMBER=8
TRAING_EPOCH=15
# MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path
MODEL_PATH=/mnt/vlm-ks3/fangyiyang/model/Qwen2.5-VL-3B-Instruct
EXPERIMENT_NAME=qwen2_5_vl_3b_emoset_${ROLLOUT_NUMBER}_grpo_emo2stage_old
CHECKPONT_PATH=/mnt/vlm-ks3/fangyiyang/emo_ckpt/${EXPERIMENT_NAME}/

python3 -m verl.trainer.main \
    config=examples/fyy/config.yaml \
    data.train_files=fuyyy74/EmoSet3steps2k@train \
    data.val_files=fuyyy74/EmoSet3steps2k@test \
    data.format_prompt=./examples/format_prompt/emo.jinja \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.n=${ROLLOUT_NUMBER} \
    worker.rollout.logprobs=null \
    worker.reward.reward_function=./examples/reward_function/emo.py:compute_score \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=${TRAING_EPOCH} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.save_checkpoint_path=${CHECKPONT_PATH}


