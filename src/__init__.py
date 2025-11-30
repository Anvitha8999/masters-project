"""
src module initialization
"""

from .model import EmotionRecognitionModel
from .data_preprocessing import MELDDataProcessor, TextProcessor, AudioProcessor
from .dataset import MELDDataset, create_dataloaders

__all__ = [
    'EmotionRecognitionModel',
    'MELDDataProcessor',
    'TextProcessor',
    'AudioProcessor',
    'MELDDataset',
    'create_dataloaders'
]
