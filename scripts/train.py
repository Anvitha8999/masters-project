

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModel
import time

print("="*60)
print("HIGH ACCURACY EMOTION RECOGNITION TRAINING")
print("="*60)
print("Using: BERT + Custom Classifier")

# Load cleaned datasets
print("Loading cleaned datasets...")
train_df = pd.read_csv("data/meld/text/train_clean.csv")
test_df = pd.read_csv("data/meld/text/test_clean.csv")

print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

# Emotion mapping
emotion_map = {
    'anger': 0, 'disgust': 1, 'fear': 2,
    'joy': 3, 'neutral': 4, 'sadness': 5, 'surprise': 6
}
id2emotion = {v: k for k, v in emotion_map.items()}

print(f"\nEmotion distribution (train):")
print(train_df['Emotion'].value_counts())

# Initialize BERT tokenizer and model
print("\nLoading BERT model...")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert_model = AutoModel.from_pretrained('bert-base-uncased')

# Freeze BERT parameters for faster training
for param in bert_model.parameters():
    param.requires_grad = False

print("BERT loaded (frozen for speed)")

# Custom Dataset
class EmotionDataset(Dataset):
    def __init__(self, texts, emotions, tokenizer, max_length=64):
        self.texts = texts
        self.emotions = emotions
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        emotion = self.emotions[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(emotion, dtype=torch.long)
        }

# Prepare data
train_texts = train_df['Utterance'].tolist()
train_emotions = train_df['Emotion'].map(emotion_map).tolist()

test_texts = test_df['Utterance'].tolist()
test_emotions = test_df['Emotion'].map(emotion_map).tolist()

train_dataset = EmotionDataset(train_texts, train_emotions, tokenizer)
test_dataset = EmotionDataset(test_texts, test_emotions, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"\nDataLoaders created:")
print(f"  Train batches: {len(train_loader)}")
print(f"  Test batches: {len(test_loader)}")

# Custom Classifier
class EmotionClassifier(nn.Module):
    def __init__(self, bert_model, num_emotions=7):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_emotions)
        self.relu = nn.ReLU()
    
    def forward(self, input_ids, attention_mask):
        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # Custom classifier
        x = self.dropout(pooled_output)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

model = EmotionClassifier(bert_model).to(device)

# Count trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nModel Statistics:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  BERT frozen: {total_params - trainable_params:,} params")

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

num_epochs = 50
best_f1 = 0
patience = 0
max_patience = 7

print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

start_time = time.time()

for epoch in range(num_epochs):
    epoch_start = time.time()
    
    # Training
    model.train()
    train_loss = 0
    train_preds = []
    train_labels = []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        train_preds.extend(preds)
        train_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    train_loss /= len(train_loader)
    train_acc = accuracy_score(train_labels, train_preds)
    train_f1 = f1_score(train_labels, train_preds, average='weighted')
    
    # Validation
    model.eval()
    test_loss = 0
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    test_loss /= len(test_loader)
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='weighted')
    
    epoch_time = time.time() - epoch_start
    total_time = time.time() - start_time
    
    print(f"\nEpoch {epoch+1}/{num_epochs} - Time: {epoch_time:.1f}s | Total: {total_time/60:.1f}min")
    print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
    print(f"Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
    
    # Learning rate scheduling
    scheduler.step(test_f1)
    
    # Save best model
    if test_f1 > best_f1:
        best_f1 = test_f1
        patience = 0
        
        # Save model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'test_f1': test_f1,
            'test_acc': test_acc,
        }, 'models/best_model.pt')
        
        print(f"\n{'='*60}")
        print(f"✓ NEW BEST MODEL! F1: {best_f1:.4f} | Acc: {test_acc:.4f}")
        print(f"{'='*60}")
        print("\nClassification Report:")
        print(classification_report(test_labels, test_preds, 
                                   target_names=list(emotion_map.keys()),
                                   digits=4))
    else:
        patience += 1
        print(f"Patience: {patience}/{max_patience}")
    
    # Early stopping
    if patience >= max_patience:
        print(f"\nEarly stopping triggered after {epoch+1} epochs")
        break

total_time = time.time() - start_time

print("\n" + "="*60)
print("TRAINING COMPLETED!")
print("="*60)
print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
print(f"Best Test F1: {best_f1:.4f}")
print(f"Best Test Accuracy: {test_acc:.4f}")
print(f"\nModel saved to: models/best_model.pt")
print("="*60)
