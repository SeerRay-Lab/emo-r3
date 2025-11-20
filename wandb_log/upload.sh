#!/bin/bash
# ============================================
# 功能：自动同步所有离线的 W&B 实验至云端
# 作者：Your Name
# ============================================

# 可选：自动登录（若环境未缓存登录）
wandb login 0aedaabc5db114ea5922799978efb09dc1d52b63


# 设置项目名与团队（按需修改）
PROJECT_NAME="emoset_r1"
ENTITY_NAME="fuyyy-wuhan-university"
# 指定wandb缓存目录（默认当前目录）
WANDB_DIR="/vlm-ssd/fangyiyang/easyr1_emo/wandb_log/qwen2_5_vl_3b_emoset_8_grpo_emothink_rethink_sft_100"








# 检查wandb目录是否存在
if [ ! -d "$WANDB_DIR" ]; then
  echo "❌ 未找到 wandb 目录: $WANDB_DIR"
  exit 1
fi

echo "🚀 开始同步 W&B 离线实验..."

# 遍历所有离线run目录
for run_path in $(find "$WANDB_DIR" -type d -name "offline-run-*"); do
  echo "🟡 正在同步: $run_path"
  wandb sync "$run_path" --project "$PROJECT_NAME" --entity "$ENTITY_NAME"
done

echo "结束。"
