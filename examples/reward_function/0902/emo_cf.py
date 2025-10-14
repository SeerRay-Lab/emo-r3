import re
from typing import Any, Dict, List


# Updated format pattern to match the template tags
FORMAT_RE = re.compile(
    r".*<coursed_thought>.*</coursed_thought>.*"
    r"<coursed_answer>\s*(positive|negative)\s*</coursed_answer>.*"
    r"<fined_thought>.*</fined_thought>.*"
    r"<fined_answer>\s*(anger|disgust|fear|joy|sadness|surprise)\s*</fined_answer>.*",
    re.DOTALL | re.IGNORECASE
)

def format_reward(response: str) -> float:
    return 1.0 if re.fullmatch(FORMAT_RE, response) else 0.0

# Updated extractors to match the new tags
TONE_RE = re.compile(r"<coursed_answer>\s*(positive|negative)\s*</coursed_answer>", re.DOTALL | re.IGNORECASE)
EMO_RE  = re.compile(r"<fined_answer>\s*(anger|disgust|fear|joy|sadness|surprise)\s*</fined_answer>", re.DOTALL | re.IGNORECASE)

def extract_tone_label(response: str) -> str:
    m = TONE_RE.search(response)
    return m.group(1).lower() if m else ""

def extract_emotion_label(response: str) -> str:
    m = EMO_RE.search(response)
    return m.group(1).lower() if m else ""


def tone_accuracy_reward(response: str, ground_truth_coarse: str) -> float:
    """Evaluate tone (coarse-grained) answer accuracy."""
    extracted_tone = extract_tone_label(response)
    # Normalize ground truth coarse label
    gt_tone = ground_truth_coarse.lower().strip()
    return 1.0 if extracted_tone == gt_tone else 0.0


def emotion_accuracy_reward(response: str, ground_truth_fine: str) -> float:
    """Evaluate emotion (fine-grained) answer accuracy."""
    extracted_emotion = extract_emotion_label(response)
    # Simple exact match for now - you may want to implement more sophisticated matching
    return 1.0 if extracted_emotion == ground_truth_fine.lower().strip() else 0.0


def compute_score(
    reward_inputs: List[Dict[str, Any]], 
    format_weight: float = 0.1,
    tone_weight: float = 0.05,
    emotion_weight: float = 0.85
) -> List[Dict[str, float]]:
    """
    Compute reward scores for emotional responses.
    
    Args:
        reward_inputs: List of dicts containing response and ground truth data
        format_weight: Weight for format compliance (default 0.1)
        tone_weight: Weight for tone accuracy (default 0.1)
        emotion_weight: Weight for emotion accuracy (default 0.8)
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for emo reward function.")
    
    # Validate weights sum to 1.0
    total_weight = format_weight + tone_weight + emotion_weight
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
    
    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # handle qwen2.5vl-32b format
        
        # Calculate individual scores
        format_score = format_reward(response)
        
        # Get tone score if coarse ground truth is available
        if "ground_truth_coarse" in reward_input:
            tone_score = tone_accuracy_reward(response, reward_input["ground_truth_coarse"])
        else:
            # If no coarse ground truth, give 0 score
            tone_score = 0.0
        
        # Get emotion score
        emotion_score = emotion_accuracy_reward(response, reward_input["ground_truth"])
        
        # Calculate weighted overall score
        overall_score = (
            format_weight * format_score + 
            tone_weight * tone_score + 
            emotion_weight * emotion_score
        )
        
        scores.append({
            "overall": overall_score,
            "format": format_score,
            "tone_accuracy": tone_score,
            "emotion_accuracy": emotion_score,
        })
    
    return scores