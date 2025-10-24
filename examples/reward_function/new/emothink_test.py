# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any, Dict, List
from mathruler.grader import grade_answer
from sentence_transformers import SentenceTransformer, util
from PIL import Image
from typing import Dict, Any, List


_clip_model = None

def _ensure_clip_model():
    global _clip_model
    if _clip_model is None:
        _clip_model = SentenceTransformer("clip-ViT-B-32")  # 或 clip-ViT-L/14 等更强模型

def _to_pil_image(img_obj):
    """
    将输入统一为 PIL.Image (如果已经是则直接返回)。
    """
    if isinstance(img_obj, Image.Image):
        return img_obj
    try:
        # 尝试从路径加载
        return Image.open(img_obj).convert("RGB")
    except Exception:
        return None


def format_reward(response: str) -> float:
    """
    Check if response follows the Affective Chain-of-Thought format:
    <step1>...</step1><step2>...</step2><step3>...</step3><final_emotion>...</final_emotion>
    """
    pattern = re.compile(
        r"<step1>.*?</step1>.*?<step2>.*?</step2>.*?<step3>.*?</step3>.*?<final_emotion>.*?</final_emotion>",
        re.DOTALL | re.IGNORECASE,
    )
    return 1.0 if re.fullmatch(pattern, response.strip()) else 0.0


def extract_emotion_content(response: str) -> Dict[str, str]:
    """
    Extract content from each tag for scoring.
    """
    tags = ["step1", "step2", "step3", "final_emotion"]
    extracted = {}
    for tag in tags:
        match = re.search(f"<{tag}>(.*?)</{tag}>", response, re.DOTALL | re.IGNORECASE)
        extracted[tag] = match.group(1).strip() if match else ""
    return extracted


def accuracy_reward(response: str, ground_truth: str) -> float:
    """
    Check if final emotion matches the ground truth emotion label (case-insensitive).
    """
    steps = extract_emotion_content(response)
    predicted = steps.get("final_emotion", "").strip().lower()
    truth = ground_truth.strip().lower()
    return 1.0 if grade_answer(predicted, truth) or predicted == truth else 0.0

def coherence_reward(
    steps: Dict[str, str],
    images: List[Any] = None,
    threshold: float = 0.5,
    env_prefix: str = "This is describing the environment and emotional atmosphere: ",
    subj_prefix: str = "This describes the main subject or person and their emotional state: "
) -> float:
    """
    图文对齐一致性（最终二元评分）：
    - Step1 与 图像环境一致性
    - Step2 与 图像主体一致性
    - coherence = (sim_env + sim_subj) / 2
    - ≥ threshold 则返回 1，否则返回 0
    """
    # 获取文本
    step1 = steps.get("step1", "").strip()
    step2 = steps.get("step2", "").strip()

    # 核心步骤缺失则直接返回 0
    if not step1 or not step2:
        return 0.0

    # 检查图像有效性
    if not images or len(images) == 0:
        print("Warning: no valid image provided, coherence defaults to 0")
        return 0.0

    # 确保 CLIP 模型加载
    _ensure_clip_model()

    # 处理图像列表为 PIL 格式
    pil_images = []
    for img in images:
        pil_img = _to_pil_image(img)
        if pil_img:
            pil_images.append(pil_img)
    if not pil_images:
        print("Warning: cannot decode images, coherence defaults to 0")
        return 0.0

    # 编码图像 embeddings
    img_embs = _clip_model.encode(pil_images, convert_to_tensor=True)
    img_emb = img_embs.mean(dim=0, keepdim=True)  # 多张图像取平均

    # 准备文本（添加语义前缀作为引导）
    step1_text = env_prefix + step1
    step2_text = subj_prefix + step2

    text_embs = _clip_model.encode([step1_text, step2_text], convert_to_tensor=True)
    env_emb = text_embs[0]
    subj_emb = text_embs[1]

    # 计算 cosine similarity
    sim_env = float(util.cos_sim(env_emb, img_emb[0]))
    sim_subj = float(util.cos_sim(subj_emb, img_emb[0]))
    avg_sim = (sim_env + sim_subj) / 2.0

    # print(f"[Coherence Debug] env_sim={sim_env:.4f}, subj_sim={sim_subj:.4f}, avg={avg_sim:.4f}")

    # 返回二元奖励
    return avg_sim


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    format_weight: float = 0.1,
    coherence_weight: float = 0.1,
) -> List[Dict[str, float]]:
    """
    Compute overall reward for Affective Chain-of-Thought outputs.
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for emotion reward function.")

    scores = []
    for reward_input in reward_inputs:
        # print(f"image: {reward_input['images']}")
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # cleanup model output
        format_score = format_reward(response)
        steps = extract_emotion_content(response)
        coherence_score = coherence_reward(steps, images=reward_input["images"])
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])

        overall = (
            format_weight * format_score
            + coherence_weight * coherence_score
            + (1 - format_weight - coherence_weight) * accuracy_score
        )

        scores.append(
            {
                "overall": overall,
                "format": format_score,
                "coherence": coherence_score,
                "accuracy": accuracy_score,
            }
        )

    return scores
