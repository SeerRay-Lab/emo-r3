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

# 初始化语义模型（可缓存）
_model = SentenceTransformer("all-MiniLM-L6-v2")


def format_reward(response: str) -> float:
    """
    Check if response follows the Affective Chain-of-Thought format:
    <step1>...</step1><step2>...</step2><step3>...</step3><final_emotion>...</final_emotion>
    """
    # pattern = re.compile(
    #     r"<step1>.*?</step1>.*?<step2>.*?</step2>.*?<step3>.*?</step3>.*?<final_emotion>.*?</final_emotion>",
    #     re.DOTALL | re.IGNORECASE,
    # )
    pattern = re.compile(
        r"<step1>[A-Za-z0-9].*?</step1>"
        r".*?<step2>[A-Za-z0-9].*?</step2>"
        r".*?<step3>[A-Za-z0-9].*?</step3>"
        r".*?<final_emotion>[A-Za-z0-9].*?</final_emotion>",
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


def coherence_reward(steps: Dict[str, str]) -> float:
    """
    Compute emotional coherence between steps.
    Uses cosine similarity of embeddings between step1-step2-step3-final_emotion.
    """
    texts = [steps[k] for k in ["step1", "step2", "step3", "final_emotion"] if steps[k]]
    if len(texts) < 2:
        return 0.0

    embeddings = _model.encode(texts, convert_to_tensor=True)
    sims = [float(util.cos_sim(embeddings[i], embeddings[i + 1])) for i in range(len(embeddings) - 1)]
    # average similarity across steps
    return sum(sims) / len(sims)


def accuracy_reward(response: str, ground_truth: str) -> float:
    """
    Check if final emotion matches the ground truth emotion label (case-insensitive).
    """
    steps = extract_emotion_content(response)
    predicted = steps.get("final_emotion", "").strip().lower()
    truth = ground_truth.strip().lower()
    return 1.0 if grade_answer(predicted, truth) or predicted == truth else 0.0


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    format_weight: float = 0.1,
    coherence_weight: float = 0.0,
) -> List[Dict[str, float]]:
    """
    Compute overall reward for Affective Chain-of-Thought outputs.
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for emotion reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # cleanup model output
        format_score = format_reward(response)
        steps = extract_emotion_content(response)
        coherence_score = coherence_reward(steps)
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
