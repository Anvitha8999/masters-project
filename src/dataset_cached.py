"""
Cached PyTorch Dataset for MELD Emotion Recognition
Processes and caches all samples once for much faster training.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm


class CachedMELDDataset(Dataset):
    """Cached PyTorch Dataset that preprocesses all data once."""
    
    def __init__(self, csv_path, processor, cache_dir=None, split='train'):
        """
        Initialize dataset with caching.
        
        Args:
            csv_path: Path to CSV file
            processor: Data processor instance
            cache_dir: Directory to store cached data
            split: 'train' or 'test'
        """
        self.processor = processor
        self.data = pd.read_csv(csv_path)
        self.split = split
        
        # Setup cache
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_file = self.cache_dir / f"{split}_cached.pkl"
        else:
            self.cache_file = None
            
        # Load or create cache
        self.cached_data = self._load_or_create_cache()
        
        print(f"✓ Loaded {len(self.cached_data)} cached samples for {split}")
    
    def _load_or_create_cache(self):
        """Load cached data or create it if it doesn't exist."""
        
        # Try to load existing cache
        if self.cache_file and self.cache_file.exists():
            print(f"Loading cached data from {self.cache_file}...")
            with open(self.cache_file, 'rb') as f:
                cached_data = pickle.load(f)
            print(f"✓ Cache loaded successfully!")
            return cached_data
        
        # Create new cache
        print(f"Creating cache for {len(self.data)} samples...")
        cached_data = []
        
        for idx in tqdm(range(len(self.data)), desc=f"Processing {self.split}"):
            row = self.data.iloc[idx]
            sample = self.processor.process_sample(row)
            cached_data.append(sample)
        
        # Save cache
        if self.cache_file:
            print(f"Saving cache to {self.cache_file}...")
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
            print(f"✓ Cache saved successfully!")
        
        return cached_data
    
    def __len__(self):
        """Return dataset size."""
        return len(self.cached_data)
    
    def __getitem__(self, idx):
        """
        Get a pre-processed sample from cache.
        
        Args:
            idx: Sample index
            
        Returns:
            dict: Cached processed sample
        """
        return self.cached_data[idx]


def create_cached_dataloaders(config, processor):
    """
    Create train and test dataloaders with caching.
    
    Args:
        config: Configuration dictionary
        processor: Data processor instance
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    
    # Get cache directory
    cache_enabled = config.get('optimization', {}).get('cache_datasets', False)
    cache_dir = config.get('optimization', {}).get('cache_dir', 'data/cache') if cache_enabled else None
    
    # Create datasets
    train_dataset = CachedMELDDataset(
        config['data']['train_csv'], 
        processor, 
        cache_dir=cache_dir,
        split='train'
    )
    test_dataset = CachedMELDDataset(
        config['data']['test_csv'], 
        processor, 
        cache_dir=cache_dir,
        split='test'
    )
    
    # Get optimization settings
    num_workers = config['training'].get('num_workers', 0)
    pin_memory = config['training'].get('pin_memory', False)
    prefetch_factor = config['training'].get('prefetch_factor', 2) if num_workers > 0 else None
    
    # Create dataloaders
    train_loader_kwargs = {
        'batch_size': config['training']['batch_size'],
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    
    # Add prefetch_factor only if num_workers > 0
    if num_workers > 0 and prefetch_factor is not None:
        train_loader_kwargs['prefetch_factor'] = prefetch_factor
    
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    
    test_loader_kwargs = {
        'batch_size': config['training']['batch_size'],
        'shuffle': False,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
    }
    
    if num_workers > 0 and prefetch_factor is not None:
        test_loader_kwargs['prefetch_factor'] = prefetch_factor
    
    test_loader = DataLoader(test_dataset, **test_loader_kwargs)
    
    return train_loader, test_loader
