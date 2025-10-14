#!/usr/bin/env python3
"""Test to clarify tokenization details"""

from transformers import AutoTokenizer

def test_tokenization_details():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    print("Understanding GPT-2 tokenization behavior:\n")
    print("=" * 80)
    
    # Test 1: Show how the space is handled
    test_cases = [
        ("Answer:", "Just 'Answer:'"),
        ("Answer: ", "'Answer:' with space"),
        ("Answer: contentment", "Full phrase"),
    ]
    
    for text, desc in test_cases:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_strings = [tokenizer.decode([t]) for t in tokens]
        
        print(f"\n{desc}:")
        print(f"Text: '{text}'")
        print(f"Tokens: {tokens}")
        print(f"Token strings: {token_strings}")
        
        # Show each token individually
        print("Individual token decoding:")
        for i, token in enumerate(tokens):
            decoded = tokenizer.decode([token])
            print(f"  Token {token} → '{decoded}'")
    
    print("\n" + "=" * 80)
    print("\nKey insight about 'contentment' tokenization:")
    
    # Test contentment in different contexts
    contentment_tests = [
        "contentment",
        " contentment",
        "Answer: contentment",
    ]
    
    for text in contentment_tests:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        token_strings = [tokenizer.decode([t]) for t in tokens]
        print(f"\nText: '{text}'")
        print(f"Tokens: {tokens} → {token_strings}")
    
    print("\n" + "=" * 80)
    print("\nExplaining the space behavior:")
    
    # Show how GPT-2 handles word boundaries
    print("\nGPT-2 uses 'Ġ' (U+0120) to represent spaces at the beginning of tokens.")
    print("When decoded, this appears as a regular space.")
    
    # Demonstrate with raw vocabulary
    print("\nChecking raw vocabulary entries:")
    vocab = tokenizer.get_vocab()
    
    # Find tokens related to our examples
    relevant_tokens = {
        220: tokenizer.decode([220]),
        2695: tokenizer.decode([2695]),
        11299: tokenizer.decode([11299]),
    }
    
    print("\nToken 220 (the space after ':'):", repr(relevant_tokens[220]))
    print("Token 2695 (in 'Answer: contentment'):", repr(relevant_tokens[2695]))
    print("Token 11299 (standalone 'content'):", repr(relevant_tokens[11299]))
    
    # Show the actual token in vocabulary
    for token_str, token_id in vocab.items():
        if token_id == 220:
            print(f"\nToken 220 in vocab: '{token_str}' (repr: {repr(token_str)})")
        if token_id == 2695:
            print(f"Token 2695 in vocab: '{token_str}' (repr: {repr(token_str)})")

if __name__ == "__main__":
    test_tokenization_details()