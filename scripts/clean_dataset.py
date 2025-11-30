"""
Clean MELD dataset by removing unnecessary columns.

This script processes the train and test CSV files and keeps only the essential columns:
- Sr No.
- Utterance
- Emotion
- Sentiment
"""

import os
import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def clean_dataset(input_file, output_file):
    """
    Clean dataset by keeping only essential columns.
    
    Args:
        input_file (Path): Path to input CSV file
        output_file (Path): Path to output CSV file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read the dataset
        print(f"Reading {input_file.name}...")
        df = pd.read_csv(input_file)
        
        # Display original columns
        print(f"Original columns: {list(df.columns)}")
        print(f"Original shape: {df.shape}")
        
        # Keep only essential columns
        essential_columns = ['Sr No.', 'Utterance', 'Emotion', 'Sentiment']
        
        # Check if all essential columns exist
        missing_cols = [col for col in essential_columns if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing columns {missing_cols}")
            return False
        
        # Select only essential columns
        df_clean = df[essential_columns]
        
        # Save cleaned dataset
        df_clean.to_csv(output_file, index=False)
        
        print(f"Cleaned columns: {list(df_clean.columns)}")
        print(f"Cleaned shape: {df_clean.shape}")
        print(f"Saved to: {output_file}")
        print("-" * 60)
        
        return True
        
    except Exception as e:
        print(f"Error processing {input_file.name}: {str(e)}")
        return False


def main():
    """Main function to clean train and test datasets."""
    
    # Define paths
    text_dir = project_root / "data" / "meld" / "text"
    
    # Check if directory exists
    if not text_dir.exists():
        print(f"Error: Directory not found at {text_dir}")
        sys.exit(1)
    
    # Files to process
    files_to_clean = [
        ("train.csv", "train_clean.csv"),
        ("test.csv", "test_clean.csv")
    ]
    
    print("=" * 60)
    print("MELD Dataset Cleaning Script")
    print("=" * 60)
    print(f"Processing files in: {text_dir}")
    print("-" * 60)
    
    successful = 0
    failed = 0
    
    for input_name, output_name in files_to_clean:
        input_file = text_dir / input_name
        output_file = text_dir / output_name
        
        if not input_file.exists():
            print(f"Warning: {input_name} not found, skipping...")
            failed += 1
            continue
        
        if clean_dataset(input_file, output_file):
            successful += 1
        else:
            failed += 1
    
    # Print summary
    print("=" * 60)
    print("Cleaning Complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(files_to_clean)}")
    print("=" * 60)
    
    # Show what to do next
    print("\nCleaned files created:")
    for _, output_name in files_to_clean:
        output_path = text_dir / output_name
        if output_path.exists():
            print(f"  - {output_name}")
    
    print("\nNote: The cleaned files contain only essential columns:")
    print("  - Sr No.")
    print("  - Utterance")
    print("  - Emotion")
    print("  - Sentiment")


if __name__ == "__main__":
    main()
