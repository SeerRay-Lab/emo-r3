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

from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(response: str) -> float:
    pattern = re.compile(r"<step1>.*</step1>.*<step2>.*</step2>.*<step3>.*</step3>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def extract_step3_content(response: str):
    m = re.search(r"</step3>(.*?)\\boxed", response, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else None

def parse_dimensions(text: str):
    text_lower = text.lower()

    # --- Valence 解析 ---
    has_positive = "positive" in text_lower
    has_negative = "negative" in text_lower

    if has_positive and has_negative:
        valence = None
    elif has_positive:
        valence = "positive"
    elif has_negative:
        valence = "negative"
    else:
        valence = None  # 没匹配到任何标签

    # --- Arousal 解析 ---
    has_low = "low" in text_lower
    has_high = "high" in text_lower

    if has_low and has_high:
        arousal = None
    elif has_low:
        arousal = "low"
    elif has_high:
        arousal = "high"
    else:
        arousal = None

    return valence, arousal

emotion_map = {
    "amusement":   ("positive", {"high"}),
    "anger":       ("negative", {"high"}),
    "awe":         ("positive", {"low"}),
    "contentment": ("positive", {"low"}), 
    "disgust":     ("negative", {"high"}),
    "excitement":  ("positive", {"high"}), 
    "fear":        ("negative", {"high"}), 
    "sadness":     ("negative", {"low"}), 
}

def think_compute(pred_valence, pred_arousal, groundtruth_label):
    gt_valence, gt_arousal_set = emotion_map[groundtruth_label]
    reward = 0.0

    # valence 匹配
    if pred_valence == gt_valence:
        reward += 0.5

    # arousal 匹配（只要有交集则 +0.5）
    if pred_arousal in gt_arousal_set:
        reward += 0.5

    return reward

def think_reward(response: str, ground_truth: str) -> float:
    step3_text = extract_step3_content(response)
    if not step3_text or ground_truth not in emotion_map:
        return 0.0

    pred_valence, pred_arousal = parse_dimensions(step3_text)

    # 如果预测完全为空，则奖励为0
    if pred_valence is None and pred_arousal is None:
        return 0.0

    return think_compute(pred_valence, pred_arousal, ground_truth)

def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1, think_weight: float = 0.1) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        think_score = think_reward(response, reward_input["ground_truth"])
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score + think_weight * think_score,
                "format": format_score,
                "accuracy": accuracy_score,
                "think": think_score,
            }
        )

    return scores
