"""
Emotion Recognition Model Architecture
CNN-BiLSTM with Attention for multimodal emotion recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class AttentionLayer(nn.Module):
    """Attention mechanism for sequence data."""
    
    def __init__(self, hidden_dim):
        """
        Initialize attention layer.
        
        Args:
            hidden_dim: Hidden dimension size
        """
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    
    def forward(self, hidden_states):
        """
        Apply attention mechanism.
        
        Args:
            hidden_states: Input hidden states (batch, seq_len, hidden_dim)
            
        Returns:
            tuple: (context_vector, attention_weights)
        """
        # Compute attention scores
        attention_scores = self.attention(hidden_states)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch, seq_len, 1)
        
        # Compute weighted sum (context vector)
        context_vector = torch.sum(attention_weights * hidden_states, dim=1)  # (batch, hidden_dim)
        
        return context_vector, attention_weights.squeeze(-1)


class TextBranch(nn.Module):
    """Text processing branch using BERT + BiLSTM."""
    
    def __init__(self, config):
        """
        Initialize text branch.
        
        Args:
            config: Model configuration
        """
        super(TextBranch, self).__init__()
        
        # BERT for text embeddings
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Freeze/Unfreeze BERT based on config
        fine_tune_bert = config.get('model', {}).get('fine_tune_bert', False)
        
        if not fine_tune_bert:
            # Freeze BERT parameters for faster training
            for param in self.bert.parameters():
                param.requires_grad = False
            print(" BERT parameters: FROZEN (faster training, lower accuracy)")
        else:
            # Unfreeze BERT for fine-tuning (better accuracy, slower)
            print(" BERT parameters: TRAINABLE (slower training, better accuracy)")
        
        bert_hidden = 768  # BERT base hidden size
        
        # Bi-LSTM
        self.bilstm = nn.LSTM(
            input_size=bert_hidden,
            hidden_size=config['model']['text_bilstm_hidden'],
            num_layers=config['model']['text_bilstm_layers'],
            batch_first=True,
            bidirectional=True,
            dropout=config['model']['dropout'] if config['model']['text_bilstm_layers'] > 1 else 0
        )
        
        # Attention
        lstm_output_dim = config['model']['text_bilstm_hidden'] * 2  # Bi-directional
        self.attention = AttentionLayer(lstm_output_dim)
        
        self.output_dim = lstm_output_dim
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass for text branch.
        
        Args:
            input_ids: BERT input IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            
        Returns:
            tuple: (text_features, attention_weights)
        """
        # Get BERT embeddings (with or without gradient based on frozen state)
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = bert_output.last_hidden_state  # (batch, seq_len, 768)
        
        # BiLSTM
        lstm_out, _ = self.bilstm(embeddings)  # (batch, seq_len, hidden*2)
        
        # Attention pooling
        text_features, attn_weights = self.attention(lstm_out)
        
        return text_features, attn_weights


class AudioBranch(nn.Module):
    """Audio processing branch using CNN + BiLSTM."""
    
    def __init__(self, config):
        """
        Initialize audio branch.
        
        Args:
            config: Model configuration
        """
        super(AudioBranch, self).__init__()
        
        input_dim = config['processing']['n_mfcc']  # MFCC features
        
        # CNN layers
        self.conv1 = nn.Conv1d(input_dim, config['model']['audio_cnn_filters'], kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(config['model']['audio_cnn_filters'], config['model']['audio_cnn_filters'], kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout_cnn = nn.Dropout(config['model']['dropout'])
        
        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=config['model']['audio_cnn_filters'],
            hidden_size=config['model']['audio_bilstm_hidden'],
            num_layers=config['model']['audio_bilstm_layers'],
            batch_first=True,
            bidirectional=True,
            dropout=config['model']['dropout'] if config['model']['audio_bilstm_layers'] > 1 else 0
        )
        
        # Attention
        lstm_output_dim = config['model']['audio_bilstm_hidden'] * 2
        self.attention = AttentionLayer(lstm_output_dim)
        
        self.output_dim = lstm_output_dim
    
    def forward(self, audio_features):
        """
        Forward pass for audio branch.
        
        Args:
            audio_features: Audio MFCC features (batch, time, mfcc)
            
        Returns:
            tuple: (audio_features, attention_weights)
        """
        # CNN expects (batch, channels, time)
        x = audio_features.transpose(1, 2)  # (batch, mfcc, time)
        
        # CNN layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout_cnn(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout_cnn(x)
        
        # Transpose back for LSTM (batch, time, features)
        x = x.transpose(1, 2)
        
        # BiLSTM
        lstm_out, _ = self.bilstm(x)
        
        # Attention pooling
        audio_features, attn_weights = self.attention(lstm_out)
        
        return audio_features, attn_weights


class EmotionRecognitionModel(nn.Module):
    """
    Complete multimodal emotion recognition model.
    Combines text and audio branches with fusion layer.
    """
    
    def __init__(self, config):
        """
        Initialize emotion recognition model.
        
        Args:
            config: Model configuration
        """
        super(EmotionRecognitionModel, self).__init__()
        
        # Text and audio branches
        self.text_branch = TextBranch(config)
        self.audio_branch = AudioBranch(config)
        
        # Fusion layer
        fusion_input_dim = self.text_branch.output_dim + self.audio_branch.output_dim
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, config['model']['fusion_hidden']),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout']),
            nn.Linear(config['model']['fusion_hidden'], config['model']['fusion_hidden'] // 2),
            nn.ReLU(),
            nn.Dropout(config['model']['dropout'])
        )
        
        # Classification head
        self.classifier = nn.Linear(
            config['model']['fusion_hidden'] // 2,
            config['model']['num_emotions']
        )
    
    def forward(self, text_input_ids, text_attention_mask, audio_features):
        """
        Forward pass through the complete model.
        
        Args:
            text_input_ids: BERT input IDs
            text_attention_mask: BERT attention mask
            audio_features: Audio MFCC features
            
        Returns:
            tuple: (logits, text_attention, audio_attention)
        """
        # Process text
        text_feat, text_attn = self.text_branch(text_input_ids, text_attention_mask)
        
        # Process audio
        audio_feat, audio_attn = self.audio_branch(audio_features)
        
        # Fuse features
        combined = torch.cat([text_feat, audio_feat], dim=1)
        fused = self.fusion(combined)
        
        # Classify
        logits = self.classifier(fused)
        
        return logits, text_attn, audio_attn


def test_model():
    """Test function for the model."""
    import yaml
    
    # Load config
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = EmotionRecognitionModel(config)
    
    # Create dummy inputs
    batch_size = 4
    text_input_ids = torch.randint(0, 30000, (batch_size, 128))
    text_attention_mask = torch.ones((batch_size, 128))
    audio_features = torch.randn(batch_size, 626, 40)  # Example dimensions
    
    # Forward pass
    logits, text_attn, audio_attn = model(text_input_ids, text_attention_mask, audio_features)
    
    print(f"Model created successfully!")
    print(f"Logits shape: {logits.shape}")
    print(f"Text attention shape: {text_attn.shape}")
    print(f"Audio attention shape: {audio_attn.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


if __name__ == "__main__":
    test_model()
