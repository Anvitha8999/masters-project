"""
Pre-cache data script
Run this once to cache all preprocessed data before training.
This makes the first training run much faster!
"""

import yaml
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path.cwd()))

from src.data_preprocessing import MELDDataProcessor
from src.dataset_cached import CachedMELDDataset

def main():
    print("="*80)
    print("DATA PRE-CACHING SCRIPT")
    print("="*80)
    print("\nThis will process and cache all training and test data.")
    print("Run this once before training for maximum speed!\n")
    
    # Load config
    config_path = 'configs/config_fast_training.yaml'
    print(f"Loading config from: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize processor
    print("\nInitializing data processor...")
    processor = MELDDataProcessor(config)
    print("✓ Data processor initialized")
    
    # Get cache directory
    cache_dir = config.get('optimization', {}).get('cache_dir', 'data/cache')
    print(f"\nCache directory: {cache_dir}")
    
    # Process and cache training data
    print("\n" + "="*80)
    print("CACHING TRAINING DATA")
    print("="*80)
    train_dataset = CachedMELDDataset(
        csv_path=config['data']['train_csv'],
        processor=processor,
        cache_dir=cache_dir,
        split='train'
    )
    
    # Process and cache test data
    print("\n" + "="*80)
    print("CACHING TEST DATA")
    print("="*80)
    test_dataset = CachedMELDDataset(
        csv_path=config['data']['test_csv'],
        processor=processor,
        cache_dir=cache_dir,
        split='test'
    )
    
    # Summary
    print("\n" + "="*80)
    print("CACHING COMPLETE!")
    print("="*80)
    print(f"✓ Training samples cached: {len(train_dataset)}")
    print(f"✓ Test samples cached: {len(test_dataset)}")
    print(f"✓ Cache location: {cache_dir}")
    print("\nYou can now run train_fast.ipynb and it will be MUCH faster!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
