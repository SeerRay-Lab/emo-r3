#!/bin/bash
# To run this script, execute the following command from the /workspace/easyr1_self directory:
# bash examples/emo/new0817/naive_grpo/qwen2_5_vl_3b_naive_grpo.sh

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path

# Create timestamped folder in PDT
TIMESTAMP=$(TZ='America/Los_Angeles' date '+%Y%m%d_%H%M%S_PDT')
RUN_FOLDER="/workspace/easyr1_self/examples/emo/new0817/naive_grpo/run_${TIMESTAMP}"
mkdir -p "${RUN_FOLDER}"

# Copy this script to the run folder
cp "/workspace/easyr1_self/examples/emo/new0817/naive_grpo/qwen2_5_vl_3b_naive_grpo.sh" "${RUN_FOLDER}/"

# Copy config file to the run folder
cp "/workspace/easyr1_self/examples/emo/new0817/naive_grpo/config.yaml" "${RUN_FOLDER}/"

# Read paths from config file
CONFIG_FILE="/workspace/easyr1_self/examples/emo/new0817/naive_grpo/config.yaml"
FORMAT_PROMPT_PATH=$(grep "format_prompt:" "$CONFIG_FILE" | awk '{print $2}')
REWARD_FUNCTION_PATH=$(grep "reward_function:" "$CONFIG_FILE" | awk '{print $2}' | sed 's/:.*//')

# Copy format_prompt file referenced in config
cp "/workspace/easyr1_self/${FORMAT_PROMPT_PATH#./}" "${RUN_FOLDER}/"

# Copy reward function file referenced in config
cp "/workspace/easyr1_self/${REWARD_FUNCTION_PATH#./}" "${RUN_FOLDER}/"

echo "Created run folder: ${RUN_FOLDER}"
echo "Copied all required files to the run folder"

python3 -m verl.trainer.main \
    config=examples/emo/new0817/naive_grpo/config.yaml \
    data.train_files=fuyyy74/Emotion6_2step_1050@train \
    data.val_files=fuyyy74/Emotion6_2step_1050@test \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    trainer.experiment_name=qwen2_5_vl_3b_naive_grpo \
    trainer.n_gpus_per_node=4


