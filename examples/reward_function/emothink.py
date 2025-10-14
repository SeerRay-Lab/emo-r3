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
import math
from typing import Any, Dict, List, Optional, Tuple

# Valid emotion categories
VALID_EMOTIONS = {
    "amusement", "anger", "awe", "contentment", 
    "disgust", "excitement", "fear", "sadness"
}

VALID_VALENCES = {"positive", "negative"}


def format_reward(response: str) -> float:
    """Check if response follows the expected format with all required tags."""
    pattern = re.compile(
        r".*<coarse_thought>.*</coarse_thought>.*"
        r"<coarse_answer>\s*(positive|negative)\s*</coarse_answer>.*"
        r"<fine_thought>.*</fine_thought>.*"
        r"<final_answer>\s*(amusement|anger|awe|contentment|disgust|excitement|fear|sadness).*</final_answer>.*",
        re.DOTALL | re.IGNORECASE
    )
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def extract_valence(response: str) -> str:
    """Extract the valence (coarse answer) from response."""
    match = re.search(r"<coarse_answer>\s*(positive|negative)\s*</coarse_answer>", response, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return ""


def extract_emotion(response: str) -> str:
    """Extract the emotion category (final answer) from response."""
    match = re.search(r"<final_answer>\s*(amusement|anger|awe|contentment|disgust|excitement|fear|sadness)", response, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return ""


def valence_accuracy_reward(response: str, ground_truth_emotion: str) -> float:
    """Evaluate valence accuracy based on emotion category."""
    extracted_valence = extract_valence(response)
    
    # Map emotions to their expected valence
    positive_emotions = {"amusement", "awe", "contentment", "excitement"}
    negative_emotions = {"anger", "disgust", "fear", "sadness"}
    
    gt_emotion = ground_truth_emotion.lower().strip()
    expected_valence = "positive" if gt_emotion in positive_emotions else "negative"
    
    return 1.0 if extracted_valence == expected_valence else 0.0


def emotion_accuracy_reward(response: str, ground_truth_emotion: str) -> float:
    """Evaluate emotion category accuracy."""
    extracted_emotion = extract_emotion(response)
    gt_emotion = ground_truth_emotion.lower().strip()
    return 1.0 if extracted_emotion == gt_emotion else 0.0


def compute_log_prob_reward(
    response: str,
    ground_truth_emotion: str,
    logprobs: Optional[Dict[str, Any]] = None,
    tokenizer: Optional[Any] = None
) -> float:
    """
    Extract log probability of the first token after <final_answer> tag.

    This function finds the first token after <final_answer> and returns its log probability.
    We verify the token matches one of the 8 valid emotions by checking the prefix.

    Args:
        response: The generated response text
        ground_truth_emotion: The correct emotion label (not used for reward, but kept for compatibility)
        logprobs: Dict containing 'token_logprobs' and 'top_logprobs'
        tokenizer: The tokenizer used (to decode tokens)

    Returns:
        log P(first token after <final_answer>), or -20.0 if format is wrong or token not found
    """
    if logprobs is None or tokenizer is None:
        raise ValueError("logprobs and tokenizer are required for compute_log_prob_reward")

    # Find "<final_answer>" in the response
    final_answer_match = re.search(r"<final_answer>", response, re.IGNORECASE)
    if not final_answer_match:
        return -20.0  # Low reward if format is wrong

    # Get the generated token IDs from logprobs
    generated_token_ids = []
    if 'token_ids' in logprobs:
        generated_token_ids = logprobs['token_ids']
    else:
        # Find where the generated text starts (after <coarse_thought>)
        generated_start_marker = response.find("<coarse_thought>")
        if generated_start_marker == -1:
            return -20.0

        generated_text = response[generated_start_marker:]
        generated_token_ids = tokenizer.encode(generated_text, add_special_tokens=False)

        # Verify the length matches
        if len(generated_token_ids) != len(logprobs['token_logprobs']):
            if len(generated_token_ids) > len(logprobs['token_logprobs']):
                generated_token_ids = generated_token_ids[:len(logprobs['token_logprobs'])]

    # Search for "<final_answer>" in the generated tokens
    final_answer_token_pos = None

    for i in range(len(generated_token_ids)):
        # Check cumulative decoding up to this point
        decoded_so_far = tokenizer.decode(generated_token_ids[:i+1])

        # Look for "<final_answer>" tag
        if "<final_answer>" in decoded_so_far.lower():
            # Found it! The next token after this is what we want
            final_answer_token_pos = i
            break

    if final_answer_token_pos is None:
        return -20.0

    # The emotion token is the next token after we found "<final_answer>"
    emotion_pos_in_logprobs = final_answer_token_pos + 1

    # Check bounds
    if emotion_pos_in_logprobs >= len(logprobs['top_logprobs']):
        return -20.0

    # Get the top_logprobs at this position
    top_probs_at_pos = logprobs['top_logprobs'][emotion_pos_in_logprobs]

    # Ground truth emotion
    gt_emotion = ground_truth_emotion.lower().strip()

    # Sort by log probability (highest to lowest)
    sorted_tokens = sorted(top_probs_at_pos.items(), key=lambda x: x[1], reverse=True)

    # Search through top-k tokens and find first one that's a prefix of ground truth
    for token_id_str, log_prob in sorted_tokens:
        token_id = int(token_id_str)
        decoded_token = tokenizer.decode([token_id]).lstrip()

        # Check if this token is a prefix of the ground truth emotion
        if gt_emotion.startswith(decoded_token.lower()):
            # Found a match! Return its log probability
            return log_prob

    # None of the top-k tokens match the ground truth
    return -20.0


def compute_score(
    reward_inputs: List[Dict[str, Any]],
    format_weight: float = 0.1,
    valence_weight: float = 0.05,
    emotion_weight: float = 0.8,
    logprob_weight: float = 0.05
) -> List[Dict[str, float]]:
    """
    Compute reward scores for emotional responses with log probability component.

    Args:
        reward_inputs: List of dicts containing response and ground truth data
        format_weight: Weight for format compliance (default 0.1)
        valence_weight: Weight for valence accuracy (default 0.05)
        emotion_weight: Weight for emotion accuracy (default 0.8)
        logprob_weight: Weight for log probability reward (default 0.05)
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for emo_cf_prob reward function.")

    # Validate weights sum to 1.0
    total_weight = format_weight + valence_weight + emotion_weight + logprob_weight
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

    scores = []
    for reward_input in reward_inputs:
        response = reward_input["response"]
        ground_truth = reward_input["ground_truth"]

        # Calculate individual scores
        format_score = format_reward(response)
        valence_score = valence_accuracy_reward(response, ground_truth)
        emotion_score = emotion_accuracy_reward(response, ground_truth)

        # Calculate log probability reward
        logprobs = reward_input.get("logprobs", None)
        tokenizer = reward_input.get("tokenizer", None)
        logprob_score = compute_log_prob_reward(response, ground_truth, logprobs, tokenizer)

        # Convert log probability to probability
        # logprob_score is the raw log probability (negative value)
        # e.g., -0.1 is very confident (high prob ~0.9), -5.0 is low confidence (prob ~0.007)
        # -20.0 means format error or invalid emotion (exp(-20) ≈ 2e-9)
        import math
        confidence = math.exp(logprob_score)

        # Calculate weighted overall score
        overall_score = (
            format_weight * format_score +
            valence_weight * valence_score +
            emotion_weight * emotion_score +
            logprob_weight * confidence
        )

        scores.append({
            "overall": overall_score,
            "format": format_score,
            "coarse_accuracy": valence_score,
            "fine_accuracy": emotion_score,
            "logprob": logprob_score,
            "confidence": confidence,
        })

    return scores