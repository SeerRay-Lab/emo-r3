#!/bin/bash

FLAG_FILE="/vlm-ssd/fangyiyang/easyr1_emo/ttrain/sheep1.txt"
LP_SCRIPT="/vlm-ssd/fangyiyang/easyr1_emo/ttrain/lp1.sh"
LS_SCRIPT="/vlm-ssd/fangyiyang/easyr1_emo/ttrain/ours1.sh"




# python3 -m verl.trainer.main_ppo \
#     algorithm.adv_estimator=grpo \
#     data.train_files=$train_files \
#     data.val_files=$val_files \
#     data.train_batch_size=512 \
#     data.max_prompt_length=1024 \
#     data.max_response_length=2048 \
#     data.filter_overlong_prompts=True \
#     data.truncation='error' \
#     data.image_key=images \
#     actor_rollout_ref.model.path=$model_path \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.actor.ppo_mini_batch_size=16 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
#     actor_rollout_ref.actor.use_kl_loss=True \
#     actor_rollout_ref.actor.kl_loss_coef=0.01 \
#     actor_rollout_ref.actor.kl_loss_type=low_var_kl \
#     actor_rollout_ref.actor.entropy_coeff=0 \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.fsdp_config.param_offload=False \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
#     actor_rollout_ref.rollout.name=vllm \
#     +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
#     actor_rollout_ref.rollout.enable_chunked_prefill=False \
#     actor_rollout_ref.rollout.enforce_eager=False \
#     actor_rollout_ref.rollout.free_cache_engine=True \
#     actor_rollout_ref.rollout.n=5 \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
#     actor_rollout_ref.ref.fsdp_config.param_offload=True \
#     algorithm.use_kl_in_reward=False \
#     trainer.critic_warmup=0 \
#     trainer.logger='["console","wandb"]' \
#     trainer.project_name='verl_loragrpo' \
#     trainer.experiment_name='qwen2_5_vl_3b_grpo' \
#     trainer.n_gpus_per_node=$n_gpus_per_node \
#     trainer.nnodes=1 \
#     trainer.save_freq=20 \
#     trainer.test_freq=5 \
#     trainer.default_local_dir=$save_path \
#     trainer.total_epochs=15 2>&1 | tee $save_path/outputs.log




# echo 0 > "$FLAG_FILE"
while true; do
    FLAG=$(cat "$FLAG_FILE")
    if [ "$FLAG" -eq 1 ]; then
        bash "$LS_SCRIPT"
        echo 0 > "$FLAG_FILE"
    else
        bash "$LP_SCRIPT"
    fi
done





# python3 -m verl.trainer.main_ppo \
#     algorithm.adv_estimator=grpo \
#     data.train_files=$train_files \
#     data.val_files=$val_files \
#     data.train_batch_size=512 \
#     data.max_prompt_length=1024 \
#     data.max_response_length=2048 \
#     data.filter_overlong_prompts=True \
#     data.truncation='error' \
#     data.image_key=images \
#     actor_rollout_ref.model.path=$model_path \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.actor.ppo_mini_batch_size=16 \
#     actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
#     actor_rollout_ref.actor.use_kl_loss=True \
#     actor_rollout_ref.actor.kl_loss_coef=0.01 \
#     actor_rollout_ref.actor.kl_loss_type=low_var_kl \
#     actor_rollout_ref.actor.entropy_coeff=0 \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.fsdp_config.param_offload=False \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#     actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
#     actor_rollout_ref.rollout.name=vllm \
#     +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
#     actor_rollout_ref.rollout.enable_chunked_prefill=False \
#     actor_rollout_ref.rollout.enforce_eager=False \
#     actor_rollout_ref.rollout.free_cache_engine=True \
#     actor_rollout_ref.rollout.n=5 \
#     actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
#     actor_rollout_ref.ref.fsdp_config.param_offload=True \
#     algorithm.use_kl_in_reward=False \
#     trainer.critic_warmup=0 \
#     trainer.logger='["console","wandb"]' \
#     trainer.project_name='verl_loragrpo' \
#     trainer.experiment_name='qwen2_5_vl_3b_grpo' \
#     trainer.n_gpus_per_node=$n_gpus_per_node \
#     trainer.nnodes=1 \
#     trainer.save_freq=20 \
#     trainer.test_freq=5 \
#     trainer.default_local_dir=$save_path \
#     trainer.total_epochs=15 2>&1 | tee $save_path/outputs.log


