source /vlm-ssd/fangyiyang/venv/easyr1/bin/activate

cd /vlm-ssd/fangyiyang/easyr1_emo

export http_proxy="http://10.224.125.58:8022"
export https_proxy="http://10.224.125.58:8022"

# bash /vlm-ssd/fangyiyang/easyr1_emo/examples/fyy/qwen2_5_vl_3b_emo_prob_grpo.sh

# bash /vlm-ssd/fangyiyang/easyr1_emo/examples/fyy/qwen2_5_vl_3b_naive_grpo.sh

bash /vlm-ssd/fangyiyang/easyr1_emo/examples/fyy/qwen2_5_vl_3b_emo_2stage_grpo.sh