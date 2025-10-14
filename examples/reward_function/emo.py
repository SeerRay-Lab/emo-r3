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
        r".*<coarse_reasoning>.*</coarse_reasoning>.*"
        r"<coarse_label>\s*(positive|negative|neutral)\s*</coarse_label>.*"
        r"<detailed_reasoning>.*</detailed_reasoning>.*"
        r"<fine_label>.*</fine_label>.*",
        re.DOTALL | re.IGNORECASE
    )
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def extract_coarse_label(response: str) -> str:
    """Extract the coarse label from response."""
    match = re.search(r"<coarse_label>\s*(positive|negative|neutral)\s*</coarse_label>", response, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return ""


def extract_fine_label(response: str) -> str:
    """Extract the fine label from response."""
    match = re.search(r"<fine_label>\s*(.+?)\s*</fine_label>", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def coarse_accuracy_reward(response: str, ground_truth_coarse: str) -> float:
    """Evaluate coarse-grained answer accuracy."""
    extracted_coarse = extract_coarse_label(response)
    # Normalize ground truth coarse label
    gt_coarse = ground_truth_coarse.lower().strip()
    return 1.0 if extracted_coarse == gt_coarse else 0.0


def fine_accuracy_reward(response: str, ground_truth_fine: str) -> float:
    """Evaluate fine-grained answer accuracy."""
    extracted_fine = extract_fine_label(response)
    # Simple exact match for now - you may want to implement more sophisticated matching
    return 1.0 if extracted_fine.lower() == ground_truth_fine.lower().strip() else 0.0


def compute_score(
    reward_inputs: List[Dict[str, Any]], 
    format_weight: float = 0.1,
    coarse_weight: float = 0.1,
    fine_weight: float = 0.8
) -> List[Dict[str, float]]:
    """
    Compute reward scores for emotional responses.
    
    Args:
        reward_inputs: List of dicts containing response and ground truth data
        format_weight: Weight for format compliance (default 0.1)
        coarse_weight: Weight for coarse-grained accuracy (default 0.3)
        fine_weight: Weight for fine-grained accuracy (default 0.6)
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for emo reward function.")
    
    # Validate weights sum to 1.0
    total_weight = format_weight + coarse_weight + fine_weight
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
    
    scores = []
    for reward_input in reward_inputs:
        response = reward_input["response"]
        
        # Calculate individual scores
        format_score = format_reward(response)
        
        # Get coarse-grained score if coarse ground truth is available
        if "ground_truth_coarse" in reward_input:
            coarse_score = coarse_accuracy_reward(response, reward_input["ground_truth_coarse"])
        else:
            # If no coarse ground truth, give 0 score
            coarse_score = 0.0
        
        # Get fine-grained score
        fine_score = fine_accuracy_reward(response, reward_input["ground_truth"])
        
        # Calculate weighted overall score
        overall_score = (
            format_weight * format_score + 
            coarse_weight * coarse_score + 
            fine_weight * fine_score
        )
        
        scores.append({
            "overall": overall_score,
            "format": format_score,
            "coarse_accuracy": coarse_score,
            "fine_accuracy": fine_score,
        })
    
    return scores