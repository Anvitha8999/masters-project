"""
PyTorch Dataset for MELD Emotion Recognition
"""

import torch
from torch.utils.data import Dataset
import pandas as pd


class MELDDataset(Dataset):
    """PyTorch Dataset for MELD emotion recognition."""
    
    def __init__(self, csv_path, processor):
        """
        Initialize dataset.
        
        Args:
            csv_path: Path to CSV file
            processor: Data processor instance
        """
        self.processor = processor
        self.data = pd.read_csv(csv_path)
        
        print(f"Loaded {len(self.data)} samples from {csv_path}")
    
    def __len__(self):
        """Return dataset size."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            dict: Processed sample
        """
        row = self.data.iloc[idx]
        sample = self.processor.process_sample(row)
        return sample


def create_dataloaders(config, processor):
    """
    Create train and test dataloaders with optimization support.
    
    Args:
        config: Configuration dictionary
        processor: Data processor instance
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    
    # Create datasets
    train_dataset = MELDDataset(config['data']['train_csv'], processor)
    test_dataset = MELDDataset(config['data']['test_csv'], processor)
    
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
