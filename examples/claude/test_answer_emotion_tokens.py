#!/usr/bin/env python3
"""Test tokenization for 'Answer: [emotion]' patterns with various prefixes"""

import torch
from transformers import AutoTokenizer

def test_answer_emotion_tokens():
    # Load tokenizer (adjust model name as needed)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Define emotions to test
    emotions = [
        "amusement", "anger", "awe", "contentment", 
        "disgust", "excitement", "fear", "sadness"
    ]
    
    # Define various arbitrary sentences to prefix before "Answer: "
    prefix_sentences = [
        "",  # No prefix
        "How are you feeling today? ",
        "The movie made me feel ",
        "After reading the story, I experienced ",
        "My emotional response was ",
        "When I saw the sunset, I felt ",
        "The joke left me with a sense of ",
        "The news report triggered ",
        "Looking at the artwork evoked ",
        "This situation makes me feel ",
        "I can't help but feel ",
        "What emotion best describes your reaction? ",
        "The dominant feeling was ",
        "Categorize the following emotion: ",
        "My current state is ",
    ]
    
    print("Testing tokenization patterns with various prefixes + 'Answer: [emotion]'\n")
    print("-" * 100)
    
    # Test each prefix with each emotion
    for prefix in prefix_sentences:
        for emotion in emotions:
            full_text = f"{prefix}Answer: {emotion}"
            
            # Tokenize full text
            tokens = tokenizer.encode(full_text, add_special_tokens=False)
            
            # Find where "Answer: " starts in the token sequence
            answer_text = "Answer: "
            answer_tokens = tokenizer.encode(answer_text, add_special_tokens=False)
            
            # Find the emotion token(s)
            emotion_tokens = tokenizer.encode(emotion, add_special_tokens=False)
            
            print(f"Text: '{full_text}'")
            print(f"Total tokens: {len(tokens)}")
            
            # Try to identify where "Answer: " appears in the token sequence
            if prefix:
                prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
                print(f"  Prefix tokens ({len(prefix_tokens)}): {prefix_tokens}")
            
            print(f"  'Answer: ' tokens: {answer_tokens} -> {[tokenizer.decode([t]) for t in answer_tokens]}")
            print(f"  '{emotion}' tokens: {emotion_tokens} -> {[tokenizer.decode([t]) for t in emotion_tokens]}")
            
            # Show last few tokens (which should contain "Answer: emotion")
            if len(tokens) >= 5:
                last_tokens = tokens[-5:]
                print(f"  Last 5 tokens: {last_tokens} -> {[tokenizer.decode([t]) for t in last_tokens]}")
            else:
                print(f"  All tokens: {tokens} -> {[tokenizer.decode([t]) for t in tokens]}")
            
            print("-" * 100)
    
    # Also test just "Answer: " with each emotion (no prefix)
    print("\nTesting just 'Answer: [emotion]' patterns:")
    print("-" * 100)
    for emotion in emotions:
        text = f"Answer: {emotion}"
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_strings = [tokenizer.decode([t]) for t in tokens]
        print(f"'{text}': {tokens} -> {token_strings}")

if __name__ == "__main__":
    test_answer_emotion_tokens()