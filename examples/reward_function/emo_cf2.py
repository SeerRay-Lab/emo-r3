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


def format_reward(response: str) -> float:
    """Check if response follows the expected format with all required tags."""
    pattern = re.compile(
        r".*<valence_thought>.*</valence_thought>.*"
        r"<final_thought>.*</final_thought>.*"
        r"<valence>\s*(positive|negative)\s*</valence>.*"
        r"<answer>\s*(amusement|anger|awe|contentment|disgust|excitement|fear|sadness)\s*</answer>.*",
        re.DOTALL | re.IGNORECASE
    )
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def extract_valence_label(response: str) -> str:
    """Extract the valence label from response."""
    match = re.search(r"<valence>\s*(positive|negative)\s*</valence>", response, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return ""


def extract_emotion_label(response: str) -> str:
    """Extract the emotion label from response."""
    match = re.search(r"<answer>\s*(amusement|anger|awe|contentment|disgust|excitement|fear|sadness)\s*</answer>", response, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return ""


def valence_accuracy_reward(response: str, ground_truth_coarse: str) -> float:
    """Evaluate valence (coarse-grained) answer accuracy."""
    extracted_valence = extract_valence_label(response)
    # Normalize ground truth coarse label
    gt_valence = ground_truth_coarse.lower().strip()
    return 1.0 if extracted_valence == gt_valence else 0.0


def emotion_accuracy_reward(response: str, ground_truth_fine: str) -> float:
    """Evaluate emotion (fine-grained) answer accuracy."""
    extracted_emotion = extract_emotion_label(response)
    # Simple exact match for now - you may want to implement more sophisticated matching
    return 1.0 if extracted_emotion == ground_truth_fine.lower().strip() else 0.0


def compute_score(
    reward_inputs: List[Dict[str, Any]], 
    format_weight: float = 0.1,
    valence_weight: float = 0.05,
    emotion_weight: float = 0.85
) -> List[Dict[str, float]]:
    """
    Compute reward scores for emotional responses.
    
    Args:
        reward_inputs: List of dicts containing response and ground truth data
        format_weight: Weight for format compliance (default 0.1)
        valence_weight: Weight for valence accuracy (default 0.1)
        emotion_weight: Weight for emotion accuracy (default 0.8)
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for emo reward function.")
    
    # Validate weights sum to 1.0
    total_weight = format_weight + valence_weight + emotion_weight
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
    
    scores = []
    for reward_input in reward_inputs:
        response = reward_input["response"]
        
        # Calculate individual scores
        format_score = format_reward(response)
        
        # Get valence score if coarse ground truth is available
        if "ground_truth_coarse" in reward_input:
            valence_score = valence_accuracy_reward(response, reward_input["ground_truth_coarse"])
        else:
            # If no coarse ground truth, give 0 score
            valence_score = 0.0
        
        # Get emotion score
        emotion_score = emotion_accuracy_reward(response, reward_input["ground_truth"])
        
        # Calculate weighted overall score
        overall_score = (
            format_weight * format_score + 
            valence_weight * valence_score + 
            emotion_weight * emotion_score
        )
        
        scores.append({
            "overall": overall_score,
            "format": format_score,
            "valence_accuracy": valence_score,
            "emotion_accuracy": emotion_score,
        })
    
    return scores