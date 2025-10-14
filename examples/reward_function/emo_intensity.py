import re
from typing import Any, Dict, List


def extract_answer_from_response(response: str, answer_label: str) -> str:
    """Extract answer from response based on the label."""
    pattern = rf"<{answer_label}>(.*?)</{answer_label}>"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def format_reward(response: str) -> float:
    """Check if response contains all required tags."""
    required_tags = ["valence", "intensity", "answer"]
    for tag in required_tags:
        if f"<{tag}>" not in response or f"</{tag}>" not in response:
            return 0.0
    return 1.0


def valence_accuracy_reward(response: str, ground_truth_valence: str) -> float:
    """Compute accuracy for valence (positive/negative)."""
    pred_valence = extract_answer_from_response(response, "valence")
    if not pred_valence:
        return 0.0
    
    # Normalize valence
    pred_valence = pred_valence.lower().strip()
    ground_truth_valence = ground_truth_valence.lower().strip()
    
    # Check for match
    if pred_valence in ["positive", "negative"] and pred_valence == ground_truth_valence:
        return 1.0
    return 0.0


def intensity_accuracy_reward(response: str, ground_truth_intensity: str) -> float:
    """Compute accuracy for intensity (1-5)."""
    pred_intensity = extract_answer_from_response(response, "intensity")
    if not pred_intensity:
        return 0.0
    
    try:
        # Extract numeric intensity
        pred_intensity_num = int(pred_intensity.strip())
        ground_truth_intensity_num = int(ground_truth_intensity)
        
        # Check if valid range and match
        if 1 <= pred_intensity_num <= 5 and pred_intensity_num == ground_truth_intensity_num:
            return 1.0
    except (ValueError, TypeError):
        pass
    
    return 0.0


def fine_accuracy_reward(response: str, ground_truth_answer: str) -> float:
    """Compute accuracy for fine-grained emotion label."""
    pred_answer = extract_answer_from_response(response, "answer")
    if not pred_answer:
        return 0.0
    
    # Normalize answers
    pred_answer = pred_answer.lower().strip()
    ground_truth_answer = ground_truth_answer.lower().strip()
    
    if pred_answer == ground_truth_answer:
        return 1.0
    return 0.0


def compute_score(
    reward_inputs: List[Dict[str, Any]], 
    format_weight: float = 0.1,
    valence_weight: float = 0.1,
    intensity_weight: float = 0.1,
    fine_weight: float = 0.7
) -> List[Dict[str, float]]:
    """
    Compute reward scores for emotion intensity responses.
    
    Args:
        reward_inputs: List of dicts containing response and ground truth data
        format_weight: Weight for format compliance (default 0.1)
        valence_weight: Weight for valence accuracy (default 0.1)
        intensity_weight: Weight for intensity accuracy (default 0.1)
        fine_weight: Weight for fine-grained accuracy (default 0.7)
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for emo_intensity reward function.")
    
    # Validate weights sum to 1.0
    total_weight = format_weight + valence_weight + intensity_weight + fine_weight
    if abs(total_weight - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
    
    
    scores = []
    for reward_input in reward_inputs:
        response = reward_input["response"]
        
        # Calculate individual scores
        format_score = format_reward(response)
        
        # Get valence score if valence ground truth is available
        if "ground_truth_coarse" in reward_input and reward_input["ground_truth_coarse"]:
            valence_score = valence_accuracy_reward(response, reward_input["ground_truth_coarse"])
        else:
            valence_score = 0.0
        
        # Get intensity score if intensity ground truth is available
        if "ground_truth_intensity" in reward_input:
            gt_intensity = reward_input["ground_truth_intensity"]
            # Check if the value is actually present and not empty
            if gt_intensity and str(gt_intensity).strip():
                intensity_score = intensity_accuracy_reward(response, str(gt_intensity))
            else:
                # Empty or None intensity value
                intensity_score = 0.0
        else:
            intensity_score = 0.0
        
        # Get fine-grained score
        fine_score = fine_accuracy_reward(response, reward_input["ground_truth"])
        
        # Calculate weighted overall score
        overall_score = (
            format_weight * format_score + 
            valence_weight * valence_score + 
            intensity_weight * intensity_score +
            fine_weight * fine_score
        )
        
        scores.append({
            "overall": overall_score,
            "format": format_score,
            "valence": valence_score,
            "intensity": intensity_score,
            "fine": fine_score,
        })
    
    return scores