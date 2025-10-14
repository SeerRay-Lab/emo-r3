#!/usr/bin/env python3
"""
Script to sample the first N items from a HuggingFace dataset and upload as a new dataset.

Example usage (run from /workspace/easyr1_self):
    # Use defaults (takes first 10 from each split)
    python examples/emo/process_dataset/sample_and_upload.py
    
    # Custom number of samples
    python examples/emo/process_dataset/sample_and_upload.py --n_samples 20
    
    # Different source and target datasets
    python examples/emo/process_dataset/sample_and_upload.py \
        --source_train fuyyy74/EmoSet3steps2k@train \
        --source_test fuyyy74/EmoSet3steps2k@test \
        --target guinea-pig/EmoSet3steps600 --n_samples 600
    
    # Make the dataset private
    python examples/emo/process_dataset/sample_and_upload.py --private

Prerequisites:
    - Login to HuggingFace Hub: huggingface-cli login
    - Install datasets library: pip install datasets
"""

import argparse
from datasets import load_dataset, DatasetDict
from huggingface_hub import HfApi, create_repo


def parse_dataset_string(dataset_string):
    """Parse dataset string format: 'username/dataset@split'"""
    if '@' in dataset_string:
        dataset_path, split = dataset_string.rsplit('@', 1)
    else:
        dataset_path = dataset_string
        split = 'train'
    return dataset_path, split


def main():
    parser = argparse.ArgumentParser(description='Sample and upload dataset to HuggingFace')
    parser.add_argument('--source_train', type=str, default='fuyyy74/EmoSet3steps2k@train',
                        help='Source training dataset in format: username/dataset@split')
    parser.add_argument('--source_test', type=str, default='fuyyy74/EmoSet3steps2k@test',
                        help='Source test dataset in format: username/dataset@split')
    parser.add_argument('--target', type=str, default='guinea-pig/EmoSet3steps10',
                        help='Target dataset name on HuggingFace')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='Number of samples to take from each split')
    parser.add_argument('--private', action='store_true',
                        help='Make the uploaded dataset private')
    
    args = parser.parse_args()
    
    # Parse source dataset strings
    train_path, train_split = parse_dataset_string(args.source_train)
    test_path, test_split = parse_dataset_string(args.source_test)
    
    # Create DatasetDict to hold both splits
    dataset_dict = DatasetDict()
    
    # Process train split
    print(f"Loading train dataset: {train_path} (split: {train_split})")
    train_dataset = load_dataset(train_path, split=train_split)
    # Take first n_samples items (not random, always from the top)
    sampled_train = train_dataset.select(range(min(args.n_samples, len(train_dataset))))
    dataset_dict['train'] = sampled_train
    print(f"Sampled first {len(sampled_train)} items from train split")
    
    # Process test split
    print(f"\nLoading test dataset: {test_path} (split: {test_split})")
    test_dataset = load_dataset(test_path, split=test_split)
    # Take first n_samples items (not random, always from the top)
    sampled_test = test_dataset.select(range(min(args.n_samples, len(test_dataset))))
    dataset_dict['test'] = sampled_test
    print(f"Sampled first {len(sampled_test)} items from test split")
    
    # Show dataset info
    print(f"\nDataset features: {sampled_train.features}")
    
    # Show a sample from train if available
    if len(sampled_train) > 0:
        print("\nFirst train sample:")
        for key, value in sampled_train[0].items():
            if isinstance(value, str):
                preview = value[:100] + "..." if len(value) > 100 else value
                print(f"  {key}: {preview}")
            else:
                print(f"  {key}: {type(value).__name__}")
    
    # Show a sample from test if available
    if len(sampled_test) > 0:
        print("\nFirst test sample:")
        for key, value in sampled_test[0].items():
            if isinstance(value, str):
                preview = value[:100] + "..." if len(value) > 100 else value
                print(f"  {key}: {preview}")
            else:
                print(f"  {key}: {type(value).__name__}")
    
    # Upload to HuggingFace
    print(f"\nUploading to HuggingFace as: {args.target}")
    
    try:
        # Push the dataset dict with both splits
        dataset_dict.push_to_hub(
            args.target,
            private=args.private,
            commit_message=f"Upload first {args.n_samples} samples from {train_path} (train and test splits)"
        )
        
        print(f"Successfully uploaded dataset to: https://huggingface.co/datasets/{args.target}")
        print(f"  - Train split: {len(sampled_train)} samples")
        print(f"  - Test split: {len(sampled_test)} samples")
        
    except Exception as e:
        print(f"Error uploading dataset: {e}")
        print("\nMake sure you are logged in to HuggingFace Hub:")
        print("  Run: huggingface-cli login")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())