#!/usr/bin/env python3
"""
Script to analyze the train split of fuyyy74/EmoSet2steps2k dataset.
Checks the number of different values in 'course_grained' and 'answer' fields.

Usage:
    cd /workspace/EasyR1
    python examples/emo/analyze_dataset.py

Requirements:
    - datasets library: pip install datasets
    - pandas library: pip install pandas
    - Internet connection for downloading the dataset from Hugging Face

The script will:
1. Load the fuyyy74/EmoSet2steps2k dataset from Hugging Face
2. Analyze the 'course_grained' and 'answer' fields
3. Display unique value counts and distributions
4. Save results to 'dataset_analysis_results.txt'

Example output:
    - Number of unique values in each field
    - Distribution of values with percentages
    - Sample data preview
    - Additional statistics (null values, total samples)
"""

import os
from datasets import load_dataset
from collections import Counter
import pandas as pd

def analyze_dataset():
    """
    Load and analyze the fuyyy74/EmoSet2steps2k dataset train split.
    """
    print("Loading fuyyy74/EmoSet2steps2k dataset...")
    
    try:
        # Load the dataset from Hugging Face
        dataset = load_dataset("fuyyy74/EmoSet2steps2k", split="train")
        print(f"Dataset loaded successfully! Total samples: {len(dataset)}")
        
        # Convert to pandas DataFrame for easier analysis
        df = dataset.to_pandas()
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check if the required fields exist
        required_fields = ['course_grained', 'answer']
        missing_fields = [field for field in required_fields if field not in df.columns]
        
        if missing_fields:
            print(f"\nWarning: Missing fields: {missing_fields}")
            print("Available columns:", list(df.columns))
            return
        
        print("\n" + "="*60)
        print("ANALYSIS RESULTS")
        print("="*60)
        
        # Analyze 'course_grained' field
        print("\n1. COURSE_GRAINED FIELD ANALYSIS:")
        print("-" * 40)
        course_grained_values = df['course_grained'].value_counts()
        print(f"Number of unique values: {len(course_grained_values)}")
        print("Value distribution:")
        for value, count in course_grained_values.items():
            print(f"  '{value}': {count} samples ({count/len(df)*100:.2f}%)")
        
        # Analyze 'answer' field
        print("\n2. ANSWER FIELD ANALYSIS:")
        print("-" * 40)
        answer_values = df['answer'].value_counts()
        print(f"Number of unique values: {len(answer_values)}")
        print("Value distribution:")
        for value, count in answer_values.items():
            print(f"  '{value}': {count} samples ({count/len(df)*100:.2f}%)")
        
        # Analyze combinations between course_grained and answer
        print("\n3. COARSE-GRAINED TO ANSWER MAPPING:")
        print("-" * 40)
        
        # Create a mapping from course_grained to unique answers
        coarse_to_answer_mapping = {}
        for _, row in df.iterrows():
            coarse = row['course_grained']
            answer = row['answer']
            if coarse not in coarse_to_answer_mapping:
                coarse_to_answer_mapping[coarse] = set()
            coarse_to_answer_mapping[coarse].add(answer)
        
        # Display the mapping
        print("Mapping from course_grained to unique answers:")
        for coarse, answers in sorted(coarse_to_answer_mapping.items()):
            print(f"  '{coarse}' -> {sorted(list(answers))}")
        
        # Check for one-to-many mappings (potential issues)
        print("\n4. VALIDATION CHECK:")
        print("-" * 40)
        problematic_mappings = []
        for coarse, answers in coarse_to_answer_mapping.items():
            if len(answers) > 1:
                problematic_mappings.append((coarse, answers))
        
        if problematic_mappings:
            print("⚠️  WARNING: Found course_grained categories with multiple different answers:")
            for coarse, answers in problematic_mappings:
                print(f"  '{coarse}' has {len(answers)} different answers: {sorted(list(answers))}")
                # Show some examples
                examples = df[df['course_grained'] == coarse][['course_grained', 'answer']].drop_duplicates()
                print(f"    Examples:")
                for _, row in examples.iterrows():
                    count = len(df[(df['course_grained'] == row['course_grained']) & (df['answer'] == row['answer'])])
                    print(f"      '{row['course_grained']}' -> '{row['answer']}' ({count} samples)")
        else:
            print("✅ VALIDATION PASSED: Each course_grained category maps to exactly one answer.")
        
        # Show all unique combinations with counts
        print("\n5. ALL COMBINATIONS (with counts):")
        print("-" * 40)
        combinations = df.groupby(['course_grained', 'answer']).size().reset_index(name='count')
        combinations = combinations.sort_values(['course_grained', 'answer'])
        print("(course_grained, answer) -> count:")
        for _, row in combinations.iterrows():
            print(f"  ('{row['course_grained']}', '{row['answer']}') -> {row['count']} samples")
        
        # Additional statistics
        print("\n6. ADDITIONAL STATISTICS:")
        print("-" * 40)
        print(f"Total samples: {len(df)}")
        print(f"Null values in 'course_grained': {df['course_grained'].isnull().sum()}")
        print(f"Null values in 'answer': {df['answer'].isnull().sum()}")
        print(f"Total unique combinations: {len(combinations)}")
        
        # Show sample data
        print("\n7. SAMPLE DATA:")
        print("-" * 40)
        print(df[['course_grained', 'answer']].head(10))
        
        # Save results to a file
        output_file = "/workspace/EasyR1/examples/emo/dataset_analysis_results.txt"
        with open(output_file, 'w') as f:
            f.write("Dataset Analysis Results for fuyyy74/EmoSet2steps2k (train split)\n")
            f.write("="*70 + "\n\n")
            f.write(f"Total samples: {len(df)}\n")
            f.write(f"Dataset shape: {df.shape}\n")
            f.write(f"Columns: {list(df.columns)}\n\n")
            
            f.write("COURSE_GRAINED FIELD:\n")
            f.write(f"Unique values: {len(course_grained_values)}\n")
            f.write("Distribution:\n")
            for value, count in course_grained_values.items():
                f.write(f"  '{value}': {count} samples ({count/len(df)*100:.2f}%)\n")
            
            f.write("\nANSWER FIELD:\n")
            f.write(f"Unique values: {len(answer_values)}\n")
            f.write("Distribution:\n")
            for value, count in answer_values.items():
                f.write(f"  '{value}': {count} samples ({count/len(df)*100:.2f}%)\n")
            
            f.write("\nCOARSE-GRAINED TO ANSWER MAPPING:\n")
            for coarse, answers in sorted(coarse_to_answer_mapping.items()):
                f.write(f"  '{coarse}' -> {sorted(list(answers))}\n")
            
            f.write("\nVALIDATION CHECK:\n")
            if problematic_mappings:
                f.write("WARNING: Found course_grained categories with multiple different answers:\n")
                for coarse, answers in problematic_mappings:
                    f.write(f"  '{coarse}' has {len(answers)} different answers: {sorted(list(answers))}\n")
            else:
                f.write("VALIDATION PASSED: Each course_grained category maps to exactly one answer.\n")
            
            f.write("\nALL COMBINATIONS (with counts):\n")
            for _, row in combinations.iterrows():
                f.write(f"  ('{row['course_grained']}', '{row['answer']}') -> {row['count']} samples\n")
            
            f.write(f"\nTotal unique combinations: {len(combinations)}\n")
        
        print(f"\nResults saved to: {output_file}")
        
    except Exception as e:
        print(f"Error loading or analyzing dataset: {e}")
        print("Make sure you have internet connection and proper Hugging Face access.")

if __name__ == "__main__":
    analyze_dataset()
