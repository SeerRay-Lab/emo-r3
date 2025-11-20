#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

ROLLOUT_NUMBER=8
EXPERIMENT_NAME=qwen2_5_vl_3b_emotion6_${ROLLOUT_NUMBER}_grpo_emothink_rethink_roll8_1108

export WANDB_MODE=offline
export WANDB_API_KEY=0aedaabc5db114ea5922799978efb09dc1d52b63
export WANDB_DIR=/vlm-ssd/fangyiyang/easyr1_emo/wandb_log/${EXPERIMENT_NAME}
export WANDB_ENTITY=fuyyy-wuhan-university

# export CUDA_VISIBLE_DEVICES=0,1

export RAY_TMPDIR=/vlm-ssd/fangyiyang/tmp_ray
mkdir -p $RAY_TMPDIR


TRAING_EPOCH=30
# MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path
MODEL_PATH=/mnt/vlm-ks3/fangyiyang/model/Qwen2.5-VL-3B-Instruct
CHECKPONT_PATH=/mnt/vlm-ks3/fangyiyang/emo_ckpt/${EXPERIMENT_NAME}/
# LOAD_CHECKPONT=/mnt/vlm-ks3/fangyiyang/emo_ckpt/qwen2_5_vl_3b_emoset_8_grpo_emothink_cot/global_step_60

python3 -m verl.trainer.main \
    config=examples/fyy/config.yaml \
    data.train_files=fuyyy74/Emotion6_2step@train \
    data.val_files=fuyyy74/Emotion6_2step@test \
    data.format_prompt=./examples/format_prompt/new/emothink_cot.jinja \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.is_rethink=true \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.n=${ROLLOUT_NUMBER} \
    worker.rollout.logprobs=null \
    worker.reward.reward_function=./examples/reward_function/new/emothink_rethink_image.py:compute_score \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=${TRAING_EPOCH} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.save_checkpoint_path=${CHECKPONT_PATH}\
    trainer.project_name=emotion6_r1
    # trainer.load_checkpoint_path=${LOAD_CHECKPONT}


