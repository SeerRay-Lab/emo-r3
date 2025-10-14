
from huggingface_hub import snapshot_download



snapshot_download(repo_id="guinea-pig/llama_factory_qwen2_5_vl_3b_EmoSet3steps2k_fullsft_1epoch", 
    local_dir="/workspace/saves", 
    repo_type="model")





