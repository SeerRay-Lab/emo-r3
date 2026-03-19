source /vlm-ssd/fangyiyang/venv/autostep/bin/activate

cd /vlm-ssd/fangyiyang/EasyR1

export http_proxy="http://10.224.125.58:8022"
export https_proxy="http://10.224.125.58:8022"

# bash /vlm-ssd/fangyiyang/EasyR1/examples/qwen3_4b_naive_geo3k.sh
bash /vlm-ssd/fangyiyang/EasyR1/examples/qwen3_4b_naive_emo.sh