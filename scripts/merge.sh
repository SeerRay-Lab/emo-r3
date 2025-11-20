cd /vlm-ssd/fangyiyang/easyr1_emo

source /vlm-ssd/fangyiyang/venv/easyr1/bin/activate

export http_proxy="http://10.224.125.58:8022"
export https_proxy="http://10.224.125.58:8022"

python3 scripts/model_merger.py \
  --local_dir /mnt/vlm-ks3/fangyiyang/emo_ckpt/qwen2_5_vl_3b_emoset_8_grpo_emothink_cot/global_step_75/actor