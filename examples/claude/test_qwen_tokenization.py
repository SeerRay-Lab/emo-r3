#!/usr/bin/env python3
"""Test tokenization for 'Answer: [emotion]' patterns with Qwen2.5-VL-3B-Instruct"""

from transformers import AutoTokenizer

def test_qwen_tokenization():
    # Load Qwen tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)
    
    # Define emotions to test
    emotions = [
        "amusement", "anger", "awe", "contentment", 
        "disgust", "excitement", "fear", "sadness"
    ]
    
    print("Testing Qwen2.5-VL-3B-Instruct tokenization patterns\n")
    print("=" * 80)
    
    # Test basic "Answer: " tokenization
    print("Basic 'Answer: ' tokenization:")
    answer_variants = [
        "Answer:",
        "Answer: ",
        "Answer: amusement",
    ]
    
    for text in answer_variants:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_strings = [tokenizer.decode([t]) for t in tokens]
        print(f"\nText: '{text}'")
        print(f"Tokens: {tokens}")
        print(f"Token strings: {token_strings}")
        
        # Show individual token decoding
        for i, token in enumerate(tokens):
            decoded = tokenizer.decode([token])
            print(f"  Token {token} → '{decoded}'")
    
    print("\n" + "=" * 80)
    print("\nTesting all emotions with 'Answer: ' prefix:")
    
    for emotion in emotions:
        text = f"Answer: {emotion}"
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_strings = [tokenizer.decode([t]) for t in tokens]
        
        print(f"\n'{text}':")
        print(f"  Tokens: {tokens}")
        print(f"  Token strings: {token_strings}")
        
        # Check if emotion is a single token
        emotion_only_tokens = tokenizer.encode(emotion, add_special_tokens=False)
        print(f"  '{emotion}' alone: {emotion_only_tokens} → {[tokenizer.decode([t]) for t in emotion_only_tokens]}")
    
    print("\n" + "=" * 80)
    print("\nTesting with various prefixes:")
    
    # Test a few examples with prefixes
    prefix_examples = [
        "How are you feeling today? Answer: amusement",
        "The movie made me feel Answer: anger",
        "My emotional response was Answer: contentment",
    ]
    
    for text in prefix_examples:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        print(f"\nText: '{text}'")
        print(f"Total tokens: {len(tokens)}")
        
        # Show last 5 tokens (which should contain "Answer: emotion")
        if len(tokens) >= 5:
            last_tokens = tokens[-5:]
            last_strings = [tokenizer.decode([t]) for t in last_tokens]
            print(f"Last 5 tokens: {last_tokens} → {last_strings}")
        else:
            token_strings = [tokenizer.decode([t]) for t in tokens]
            print(f"All tokens: {tokens} → {token_strings}")
    
    print("\n" + "=" * 80)
    print("\nVocabulary size:", tokenizer.vocab_size)
    
    # Try to understand how Qwen handles spaces
    print("\nSpace handling tests:")
    space_tests = [
        " ",
        "  ",
        "word",
        " word",
        "word ",
        " word ",
    ]
    
    for text in space_tests:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_strings = [tokenizer.decode([t]) for t in tokens]
        print(f"'{text}' → {tokens} → {token_strings}")

if __name__ == "__main__":
    test_qwen_tokenization()