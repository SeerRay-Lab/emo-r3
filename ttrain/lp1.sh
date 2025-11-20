# source /vlm-ssd/fangyiyang/venv/easyr1/bin/activate
# cd /vlm-ssd/fangyiyang/easyr1_emo
# #!/bin/bash

# set -x

# export http_proxy="http://10.224.125.58:8022"
# export https_proxy="http://10.224.125.58:8022"

# export PYTHONUNBUFFERED=1

# # export WANDB_API_KEY=0aedaabc5db114ea5922799978efb09dc1d52b63
# # export WANDB_DIR=/vlm-ssd/fangyiyang/easyr1_emo/wandb_log
# # export WANDB_ENTITY=fuyyy-wuhan-university

# ROLLOUT_NUMBER=5
# TRAING_EPOCH=1
# # MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path
# MODEL_PATH=/mnt/vlm-ks3/fangyiyang/model/Qwen2.5-VL-3B-Instruct
# EXPERIMENT_NAME=qwen2_5_vl_3b_emoset_${ROLLOUT_NUMBER}_grpo_emothink
# CHECKPONT_PATH=/mnt/vlm-ks3/fangyiyang/emo_ckpt/test
# mkdir -p $CHECKPONT_PATH

# python3 -m verl.trainer.main \
#     config=examples/emo/old_pre_0818/cf_prob_grpo/config_emo_prob.yaml \
#     data.train_files=fuyyy74/EmoSet3steps2k@train \
#     data.val_files=fuyyy74/EmoSet3steps2k@test \
#     worker.actor.model.model_path=${MODEL_PATH} \
#     worker.rollout.tensor_parallel_size=1 \
#     worker.rollout.n=${ROLLOUT_NUMBER} \
#     trainer.n_gpus_per_node=8 \
#     trainer.total_epochs=${TRAING_EPOCH} \
#     trainer.experiment_name=${EXPERIMENT_NAME} \
#     trainer.save_checkpoint_path=${CHECKPONT_PATH} \
#     trainer.logger='["console"]'

# echo "Removing: $CHECKPONT_PATH"
# rm -rf $CHECKPONT_PATH

# echo "download..."
sleep 5