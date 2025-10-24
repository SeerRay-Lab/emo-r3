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

import math
from mathruler.grader import extract_boxed_content, grade_answer

import json

# Load external keyword json
with open("examples/reward_function/new/emo_token_pool.json", "r") as f:
    keywords = json.load(f)

POSITIVE_WORDS = set(keywords["valence_positive"])
NEGATIVE_WORDS = set(keywords["valence_negative"])
HIGH_AROUSAL_WORDS = set(keywords["arousal_high"])
LOW_AROUSAL_WORDS = set(keywords["arousal_low"])


GROUND_TRUTH_EMOTION_COORDS = {
    "amusement": (0.8, 0.4),
    "excitement": (0.9, 0.9),
    "contentment": (0.7, 0.2),
    "awe": (0.6, 0.7),
    "anger": (-0.8, 0.8),
    "fear": (-0.9, 0.9),
    "disgust": (-0.7, 0.5),
    "sadness": (-0.9, 0.2)
}


def format_reward(response: str) -> float:
    pattern = re.compile(r"<step1>.*</step1>.*<step2>.*</step2>.*<step3>.*</step3>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def normalize_valence(score, scale=0.5):
    """Normalize to [-1, 1] with tanh to prevent spamming."""
    return math.tanh(score * scale)


def normalize_arousal(score, scale=0.5):
    """Return arousal in [0, 1] using sigmoid."""
    return 1 / (1 + math.exp(-score * scale))

def extract_valence_arousal(think_text):
    valence_score = 0
    arousal_score = 0
    words = think_text.lower().split()

    for word in words:
        if word in POSITIVE_WORDS:
            valence_score += 1
        elif word in NEGATIVE_WORDS:
            valence_score -= 1
        if word in HIGH_AROUSAL_WORDS:
            arousal_score += 1
        elif word in LOW_AROUSAL_WORDS:
            arousal_score -= 1

    valence = normalize_valence(valence_score)
    arousal = normalize_arousal(arousal_score)
    return (valence, arousal)

def remove_boxed(text):
    # Pattern to match \boxed{...}
    pattern = re.compile(r"\\boxed\s*\{.*?\}", re.DOTALL)
    cleaned_text = re.sub(pattern, "", text)
    return cleaned_text.strip()

def emo_reward(response: str, ground_truth: str):
    think_text = remove_boxed(response)
    pred_coord = extract_valence_arousal(think_text)
    true_coord = GROUND_TRUTH_EMOTION_COORDS[ground_truth]

    distance = ((pred_coord[0] - true_coord[0])**2 + 4 * (pred_coord[1] - true_coord[1])**2) / 8
    VA_reward = 1 - distance

    # valence_match = (pred_coord[0] * true_coord[0] > 0)
    # arousal_match = (pred_coord[1] * true_coord[1] > 0)
    # think_reward = 0
    # think_reward += 0.5 if valence_match else -0.5
    # think_reward += 0.5 if arousal_match else -0.5

    return VA_reward



def compute_score(reward_inputs: List[Dict[str, Any]], format_weight: float = 0.1, think_weight: float = 0.1) -> List[Dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        format_score = format_reward(response)
        accuracy_score = accuracy_reward(response, reward_input["ground_truth"])
        emo_score = emo_reward(response, reward_input["ground_truth"])
        scores.append(
            {
                "overall": (1 - format_weight - think_weight) * accuracy_score + format_weight * format_score + think_weight * emo_score,
                "format": format_score,
                "accuracy": accuracy_score,
                "think": emo_score,
            }
        )

    return scores
