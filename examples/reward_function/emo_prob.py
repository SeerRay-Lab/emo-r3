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
        r".*<valence_thought>.*</valence_thought>.*"
        r"<final_thought>.*</final_thought>.*"
        r"<valence>\s*(positive|negative)\s*</valence>.*"
        r"Answer: (amusement|anger|awe|contentment|disgust|excitement|fear|sadness).*",
        re.DOTALL | re.IGNORECASE
    )
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def extract_valence(response: str) -> str:
    """Extract the valence from response."""
    match = re.search(r"<valence>\s*(positive|negative)\s*</valence>", response, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return ""


def extract_emotion(response: str) -> str:
    """Extract the emotion category from response."""
    match = re.search(r"Answer: (amusement|anger|awe|contentment|disgust|excitement|fear|sadness)", response, re.IGNORECASE)
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
    Compute -log P(correct emotion) reward after softmax over all 8 emotions.
    
    Args:
        response: The generated response text
        ground_truth_emotion: The correct emotion label
        logprobs: Dict containing 'token_logprobs' and 'top_logprobs'
        tokenizer: The tokenizer used (to get emotion token IDs)
    
    Returns:
        -log P(correct emotion) after softmax normalization
    """
    if logprobs is None or tokenizer is None:
        raise ValueError("logprobs and tokenizer are required for compute_log_prob_reward")
    
    # Debug: Check logprobs structure
    # print(f"\n=== Debug: Logprobs structure ===")
    # print(f"Type of logprobs: {type(logprobs)}")
    # if isinstance(logprobs, dict):
    #     print(f"Keys in logprobs: {list(logprobs.keys())}")
    #     if 'token_logprobs' in logprobs:
    #         print(f"Length of token_logprobs: {len(logprobs['token_logprobs'])}")
    #     if 'top_logprobs' in logprobs:
    #         print(f"Length of top_logprobs: {len(logprobs['top_logprobs'])}")
    #         if len(logprobs['top_logprobs']) > 0:
    #             print(f"Type of first top_logprobs: {type(logprobs['top_logprobs'][0])}")
    #             print(f"Number of tokens in first position: {len(logprobs['top_logprobs'][0])}")
    
    gt_emotion = ground_truth_emotion.lower().strip()
    
    # Find "Answer: " in the response (with exactly one space)
    answer_match = re.search(r"Answer: ", response, re.IGNORECASE)
    if not answer_match:
        # print(f"ERROR: 'Answer: ' not found in response: {response[:100]}...")
        return 10.0  # High penalty if format is wrong
    
    # Tokenize up to "Answer: " to find position
    prefix = response[:answer_match.end()]
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    emotion_start_pos = len(prefix_tokens)
    
    # print(f"\n=== Debug: Token position ===")
    # print(f"Prefix text: '{prefix}'")
    # print(f"Prefix tokens: {prefix_tokens}")
    # print(f"Emotion should start at position: {emotion_start_pos}")
    
    # Get token IDs for all 8 emotions with space prefix
    # Hardcode based on our tokenization tests
    emotion_to_tokens = {}
    # print(f"\n=== Debug: Emotion tokenization ===")
    for emotion in VALID_EMOTIONS:
        emotion_with_space = f" {emotion}"
        tokens = tokenizer.encode(emotion_with_space, add_special_tokens=False)
        emotion_to_tokens[emotion] = tokens
        # decoded = tokenizer.decode(tokens)
        # print(f"Emotion '{emotion}' -> tokens {tokens} -> decoded '{decoded}'")
    
    # Collect log probabilities for all 8 emotions
    emotion_logprobs = {}
    
    # Check if we have logprobs at the emotion position
    if emotion_start_pos >= len(logprobs['token_logprobs']):
        # print(f"ERROR: emotion_start_pos {emotion_start_pos} >= len(token_logprobs) {len(logprobs['token_logprobs'])}")
        return 10.0
    
    # Get top_logprobs at the first token position
    if 'top_logprobs' in logprobs and emotion_start_pos < len(logprobs['top_logprobs']):
        top_probs_at_pos = logprobs['top_logprobs'][emotion_start_pos]
    else:
        # print(f"ERROR: No top_logprobs at position {emotion_start_pos}")
        return 10.0
    
    # Debug: What was actually generated after "Answer: "?
    # print(f"\n=== Debug: What comes after 'Answer: ' ===")
    # Tokenize the full response to see what tokens follow
    full_tokens = tokenizer.encode(response, add_special_tokens=False)
    # print(f"Full response tokens (around emotion position): {full_tokens[max(0, emotion_start_pos-2):emotion_start_pos+5]}")
    if emotion_start_pos < len(full_tokens):
        next_token = full_tokens[emotion_start_pos]
        decoded_next = tokenizer.decode([next_token])
        # print(f"Token at position {emotion_start_pos}: {next_token} -> '{decoded_next}'")
    
    # Better approach: Work directly with the generated text since logprobs only contain generated tokens
    
    # Extract just the generated portion from the response
    # The response contains prompt + generated text, but we only need generated
    # We can reconstruct the generated text from token_ids that were passed along with logprobs
    
    # Get the generated token IDs from the response
    # These should match the length of logprobs
    generated_token_ids = []
    if 'token_ids' in logprobs:
        generated_token_ids = logprobs['token_ids']
    else:
        # If token_ids not passed, we need to figure out the generated portion
        # The safest way is to look for a clear delimiter in the response
        # Usually the generated part starts after the last user message or after a specific tag
        
        # For this task, we know the generated part contains "<valence>...</valence>\nAnswer: emotion"
        # Let's find where the generated text likely starts
        generated_start_marker = response.find("<valence>")
        if generated_start_marker == -1:
            # print("ERROR: Could not find generated text start marker")
            return 10.0
        
        generated_text = response[generated_start_marker:]
        generated_token_ids = tokenizer.encode(generated_text, add_special_tokens=False)
        
        # Verify the length matches
        if len(generated_token_ids) != len(logprobs['token_logprobs']):
            # print(f"WARNING: Generated tokens ({len(generated_token_ids)}) != logprobs length ({len(logprobs['token_logprobs'])})")
            # Try to adjust by looking at the end
            if len(generated_token_ids) > len(logprobs['token_logprobs']):
                # Trim from the end (might have extra tokens)
                generated_token_ids = generated_token_ids[:len(logprobs['token_logprobs'])]
    
    # print(f"Working with {len(generated_token_ids)} generated tokens")
    # print(f"Generated text preview: {tokenizer.decode(generated_token_ids[:20])[:100]}...")
    
    # Debug: Show tokens as we search for "Answer:"
    # print("\n=== Debug: Searching for 'Answer:' in generated tokens ===")
    answer_token_pos = None
    
    for i in range(len(generated_token_ids)):
        token = generated_token_ids[i]
        decoded = tokenizer.decode([token])
        # Also show cumulative decoding up to this point
        decoded_so_far = tokenizer.decode(generated_token_ids[:i+1])
        # print(f"Position {i}: token={token}, decoded='{decoded}', cumulative='{decoded_so_far[-50:]}'")
        
        # Check for "Answer:"
        if decoded.strip().lower() == "answer:":
            answer_token_pos = i
            # print(f"  -> Found 'Answer:' as single token!")
            break
        
        # Check two tokens for "Answer" + ":"
        if i < len(generated_token_ids) - 1:
            two_tokens = tokenizer.decode(generated_token_ids[i:i+2])
            if two_tokens.strip().lower() == "answer:":
                answer_token_pos = i + 1  # Position of the ":" token
                # print(f"  -> Found 'Answer:' as two tokens at positions {i}-{i+1}!")
                break
    
    if answer_token_pos is None:
        # print("ERROR: Could not find 'Answer:' in generated token sequence")
        # Show more tokens if we didn't find it in the first 20
        # print("Full generated text:", tokenizer.decode(generated_token_ids))
        return 10.0
    
    # The emotion token is the next token after "Answer:"
    emotion_pos_in_logprobs = answer_token_pos + 1
    
    # Debug: Show what's at and around the emotion position
    # print(f"\n=== Debug: Tokens around emotion position ===")
    start = max(0, emotion_pos_in_logprobs - 2)
    end = min(len(generated_token_ids), emotion_pos_in_logprobs + 3)
    
    for i in range(start, end):
        if i < len(generated_token_ids):
            token = generated_token_ids[i]
            decoded = tokenizer.decode([token])
            marker = " <-- EMOTION POSITION" if i == emotion_pos_in_logprobs else ""
            # print(f"Position {i}: token={token}, decoded='{decoded}'{marker}")
    
    # Check if next token might be a space, if so, skip it
    if emotion_pos_in_logprobs < len(generated_token_ids):
        next_token = tokenizer.decode([generated_token_ids[emotion_pos_in_logprobs]])
        if next_token.strip() == "":  # It's whitespace
            # print(f"Token at position {emotion_pos_in_logprobs} is whitespace, checking next position")
            emotion_pos_in_logprobs += 1
            # if emotion_pos_in_logprobs < len(generated_token_ids):
            #     actual_emotion_token = generated_token_ids[emotion_pos_in_logprobs]
            #     print(f"Position {emotion_pos_in_logprobs}: token={actual_emotion_token}, decoded='{tokenizer.decode([actual_emotion_token])}'")
    
    # print(f"\nFinal emotion position in logprobs: {emotion_pos_in_logprobs}")
    
    # Check bounds
    if emotion_pos_in_logprobs >= 0 and emotion_pos_in_logprobs < len(logprobs['top_logprobs']):
        top_probs_at_pos = logprobs['top_logprobs'][emotion_pos_in_logprobs]
        # print(f"Using position {emotion_pos_in_logprobs} for logprobs lookup")
    else:
        # print(f"ERROR: Position {emotion_pos_in_logprobs} is out of bounds (logprobs length: {len(logprobs['top_logprobs'])})")
        return 10.0
    
    # Debug: Print top-k tokens and their decoded text
    # print(f"\n=== Debug: Top-k tokens at position after 'Answer: ' ===")
    # print(f"Ground truth emotion: {gt_emotion}")
    # print(f"Number of top tokens: {len(top_probs_at_pos)}")
    
    # Sort by log probability (descending)
    sorted_tokens = sorted(top_probs_at_pos.items(), key=lambda x: x[1], reverse=True)
    
    for i, (token_id_str, log_prob) in enumerate(sorted_tokens[:20]):  # Show top 20
        token_id = int(token_id_str)
        decoded = tokenizer.decode([token_id])
        prob = math.exp(log_prob)
        # print(f"  #{i+1}: token_id={token_id}, decoded='{decoded}', log_prob={log_prob:.4f}, prob={prob:.4f}")
        
        # Check if this is one of our emotions
        # for emotion in VALID_EMOTIONS:
        #     if decoded.strip() == emotion:
        #         print(f"       ^ This is emotion: {emotion}")
        #         break
    
    # print(f"=== End Debug ===\n")
    
    # For each emotion, get its log probability
    # print(f"\n=== Debug: Looking for emotion tokens in top_probs ===")
    for emotion, tokens in emotion_to_tokens.items():
        if len(tokens) == 1:
            # Single token emotion (most cases)
            token_id = tokens[0]
            # print(f"\nEmotion '{emotion}' has single token: {token_id}")
            
            # Search through top_probs to find this token
            found = False
            # First try direct lookup with string key
            token_id_str = str(token_id)
            if token_id_str in top_probs_at_pos:
                emotion_logprobs[emotion] = top_probs_at_pos[token_id_str]
                found = True
                # print(f"  Found via direct lookup: logprob={top_probs_at_pos[token_id_str]}")
            else:
                # Try searching by decoding
                for tid_str, log_prob in top_probs_at_pos.items():
                    # Decode the token to check if it matches our emotion
                    decoded = tokenizer.decode([int(tid_str)])
                    if decoded.strip() == emotion:
                        emotion_logprobs[emotion] = log_prob
                        found = True
                        # print(f"  Found via decode match: token {tid_str} -> '{decoded}' logprob={log_prob}")
                        break
            
            if not found:
                # Not in top-k, assign very negative log prob
                emotion_logprobs[emotion] = -20.0
                # print(f"  NOT FOUND in top-k tokens")
        else:
            # Multi-token emotion (e.g., contentment)
            # For simplicity, check if first token matches
            first_token = tokens[0]
            # print(f"\nEmotion '{emotion}' has multiple tokens: {tokens}")
            # print(f"  Looking for first token: {first_token}")
            
            found = False
            token_id_str = str(first_token)
            if token_id_str in top_probs_at_pos:
                # Verify by decoding
                decoded = tokenizer.decode([first_token])
                # print(f"  First token {first_token} -> '{decoded}'")
                # For contentment, this would be " content"
                if emotion.startswith(decoded.strip()):
                    emotion_logprobs[emotion] = top_probs_at_pos[token_id_str]
                    found = True
                    # print(f"  Found via direct lookup: logprob={top_probs_at_pos[token_id_str]}")
            
            if not found:
                emotion_logprobs[emotion] = -20.0
                # print(f"  NOT FOUND in top-k tokens")
    
    # Apply softmax to get probability distribution
    import numpy as np
    
    # Convert to numpy array in consistent order
    emotions_list = sorted(VALID_EMOTIONS)
    logprobs_array = np.array([emotion_logprobs[e] for e in emotions_list])
    
    # Debug: Show collected emotion probabilities
    # print(f"\n=== Debug: Emotion logprobs before softmax ===")
    # for emotion in emotions_list:
    #     log_prob = emotion_logprobs[emotion]
    #     print(f"  {emotion}: log_prob={log_prob:.4f}, prob={math.exp(log_prob):.6f}")
    
    # Softmax: exp(logprob) / sum(exp(logprobs))
    # For numerical stability: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    max_logprob = np.max(logprobs_array)
    exp_probs = np.exp(logprobs_array - max_logprob)
    softmax_probs = exp_probs / np.sum(exp_probs)
    
    # Debug: Show softmax probabilities
    # print(f"\n=== Debug: After softmax normalization ===")
    # for i, emotion in enumerate(emotions_list):
    #     is_correct = " <-- CORRECT" if emotion == gt_emotion else ""
    #     print(f"  {emotion}: prob={softmax_probs[i]:.6f}{is_correct}")
    
    # Get probability of correct emotion
    correct_idx = emotions_list.index(gt_emotion)
    correct_prob = softmax_probs[correct_idx]
    
    # Debug: Show final reward
    neg_log_prob = -np.log(correct_prob + 1e-10)
    # print(f"\n=== Debug: Final reward ===")
    # print(f"  P(correct) = {correct_prob:.6f}")
    # print(f"  -log P(correct) = {neg_log_prob:.4f}")
    # print(f"=== End Debug ===\n")
    
    # Return -log P(correct)
    return neg_log_prob


def compute_score(
    reward_inputs: List[Dict[str, Any]], 
    format_weight: float = 0.05,
    valence_weight: float = 0.05,
    emotion_weight: float = 0.8,
    logprob_weight: float = 0.1
) -> List[Dict[str, float]]:
    """
    Compute reward scores for emotional responses with log probability component.
    
    Args:
        reward_inputs: List of dicts containing response and ground truth data
        format_weight: Weight for format compliance (default 0.1)
        valence_weight: Weight for valence accuracy (default 0.2)
        emotion_weight: Weight for emotion accuracy (default 0.3)
        logprob_weight: Weight for log probability reward (default 0.4)
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for emo_prob reward function.")
    
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
        
        # Normalize logprob_score to [0, 1] range
        # If logprob_score is very high (e.g., 19.9260 when all emotions are -20.0),
        # it means none of the valid emotions were in top-k, so set normalized score to 0
        if logprob_score > 10.0:  # Threshold for "all emotions missing"
            normalized_logprob_score = 0.0
        else:
            # Option 1: Sigmoid-like transformation
            # Maps: 0 -> 1.0, 2.08 (random) -> ~0.5, 5+ -> ~0
            import math
            normalized_logprob_score = math.exp(-logprob_score / 2.0)
        
        # Option 2: Linear clipping (uncomment to use)
        # max_logprob = 5.0  # Scores above this are treated as 0
        # normalized_logprob_score = max(0, 1 - logprob_score / max_logprob)
        
        # Option 3: Convert to accuracy-like score (uncomment to use)
        # threshold = 2.08  # -log(1/8) = random baseline
        # normalized_logprob_score = 1.0 if logprob_score < threshold else 0.0
        
        # Calculate weighted overall score
        overall_score = (
            format_weight * format_score + 
            valence_weight * valence_score + 
            emotion_weight * emotion_score +
            logprob_weight * normalized_logprob_score
        )
        
        scores.append({
            "overall": overall_score,
            "format": format_score,
            "valence_accuracy": valence_score,
            "emotion_accuracy": emotion_score,
            "logprob_reward": logprob_score,
            "normalized_logprob": normalized_logprob_score,
        })
    
    return scores