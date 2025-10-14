


from huggingface_hub import HfApi
api = HfApi()


api.upload_large_folder(
    repo_id="guinea-pig/qwen2_5_vl_3b_naive_grpo_0817",
    repo_type="model",
    folder_path="/workspace/easyr1_self/checkpoints",
    private=False,
)


# api.upload_folder(
#     folder_path="/workspace/easyr1_self/checkpoints",
#     repo_id="guinea-pig/qwen2_5_vl_3b_0e_20e_prob_0802_grpo",
#     repo_type="model",
#     # path_in_repo="",              # "" = root of the repo
# )




