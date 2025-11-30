"""
Data Preprocessing Module
Handles loading and preprocessing of text and audio data.
"""

import numpy as np
import pandas as pd
import librosa
import torch
from pathlib import Path
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder


class TextProcessor:
    """Processes text data using BERT tokenizer."""
    
    def __init__(self, max_length=128):
        """
        Initialize text processor.
        
        Args:
            max_length: Maximum sequence length for tokenization
        """
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def process(self, text):
        """
        Process a single text utterance.
        
        Args:
            text: Input text string
            
        Returns:
            dict: Tokenized and encoded text
        """
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0)
        }


class AudioProcessor:
    """Processes audio data extracting MFCCs and spectrograms."""
    
    def __init__(self, sample_rate=16000, max_length=10, n_mfcc=40):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Target sample rate for audio
            max_length: Maximum audio length in seconds
            n_mfcc: Number of MFCC coefficients
        """
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.n_mfcc = n_mfcc
        self.max_samples = sample_rate * max_length
    
    def load_audio(self, audio_path):
        """
        Load audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            np.array: Audio waveform
        """
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Pad or truncate to max_length
            if len(audio) > self.max_samples:
                audio = audio[:self.max_samples]
            else:
                audio = np.pad(audio, (0, self.max_samples - len(audio)))
            
            return audio
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return zeros if file not found
            return np.zeros(self.max_samples)
    
    def extract_mfcc(self, audio):
        """
        Extract MFCC features from audio.
        
        Args:
            audio: Audio waveform
            
        Returns:
            np.array: MFCC features
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=512,
            hop_length=256
        )
        
        # Transpose to (time, features)
        mfcc = mfcc.T
        
        return mfcc
    
    def process(self, audio_path):
        """
        Process a single audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            torch.Tensor: Processed audio features
        """
        # Load audio
        audio = self.load_audio(audio_path)
        
        # Extract MFCC features
        mfcc = self.extract_mfcc(audio)
        
        # Convert to tensor
        mfcc_tensor = torch.from_numpy(mfcc).float()
        
        return mfcc_tensor


class MELDDataProcessor:
    """Main data processor for MELD dataset."""
    
    def __init__(self, config):
        """
        Initialize MELD data processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.text_processor = TextProcessor(config['processing']['max_text_length'])
        self.audio_processor = AudioProcessor(
            sample_rate=config['processing']['sample_rate'],
            max_length=config['processing']['max_audio_length'],
            n_mfcc=config['processing']['n_mfcc']
        )
        
        # Label encoder for emotions
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(config['emotions']['labels'])
    
    def load_csv(self, csv_path):
        """
        Load data from CSV file.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        df = pd.read_csv(csv_path)
        return df
    
    def get_audio_path(self, dialogue_id, utterance_id):
        """
        Construct audio file path from dialogue and utterance IDs.
        
        Args:
            dialogue_id: Dialogue ID
            utterance_id: Utterance ID
            
        Returns:
            Path: Audio file path
        """
        audio_dir = Path(self.config['data']['audio_dir'])
        filename = f"dia{dialogue_id}_utt{utterance_id}.wav"
        return audio_dir / filename
    
    def process_sample(self, row):
        """
        Process a single dataapplesample.
        
        Args:
            row: DataFrame row
            
        Returns:
            dict: Processed sample with text, audio, and label
        """
        # Process text
        text_data = self.text_processor.process(str(row['Utterance']))
        
        # Get audio path and process
        audio_path = self.get_audio_path(row['Dialogue_ID'], row['Utterance_ID'])
        audio_data = self.audio_processor.process(str(audio_path))
        
        # Encode label
        label = self.label_encoder.transform([row['Emotion']])[0]
        
        return {
            'text_input_ids': text_data['input_ids'],
            'text_attention_mask': text_data['attention_mask'],
            'audio_features': audio_data,
            'label': torch.tensor(label, dtype=torch.long)
        }


def test_preprocessor():
    """Test function for the preprocessor."""
    import yaml
    
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize processor
    processor = MELDDataProcessor(config)
    
    # Load train data
    df = processor.load_csv(config['data']['train_csv'])
    
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst sample:")
    print(df.iloc[0])
    
    # Process first sample
    sample = processor.process_sample(df.iloc[0])
    print(f"\nProcessed sample shapes:")
    print(f"Text input IDs: {sample['text_input_ids'].shape}")
    print(f"Audio features: {sample['audio_features'].shape}")
    print(f"Label: {sample['label']}")


if __name__ == "__main__":
    test_preprocessor()
