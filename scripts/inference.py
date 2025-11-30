"""
Inference script for BERT-based emotion classifier
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import yaml
from pathlib import Path

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
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            pooled_output = outputs.last_hidden_state[:, 0, :]
        
        x = self.dropout(pooled_output)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class EmotionPredictor:
    def __init__(self, model_path='models/best_model.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load BERT
        print("Loading BERT...")
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        bert_model = AutoModel.from_pretrained('bert-base-uncased')
        
        # Freeze BERT
        for param in bert_model.parameters():
            param.requires_grad = False
        
        # Load trained model
        self.model = EmotionClassifier(bert_model).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from epoch {checkpoint['epoch']}")
        print(f"Test F1: {checkpoint['test_f1']:.4f}")
        print(f"Test Accuracy: {checkpoint['test_acc']:.4f}\n")
        
        self.emotion_labels = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    
    def predict(self, text):
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = torch.argmax(probs).item()
            
        return {
            'emotion': self.emotion_labels[pred_idx],
            'confidence': probs[pred_idx].item(),
            'all_probabilities': {
                label: prob.item()
                for label, prob in zip(self.emotion_labels, probs)
            }
        }


def interactive_mode():
    predictor = EmotionPredictor()
    
    print("="*60)
    print("INTERACTIVE EMOTION RECOGNITION")
    print("="*60)
    print("Enter text to analyze emotion (or 'quit' to exit)\n")
    
    while True:
        text = input("Enter text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not text:
            continue
        
        result = predictor.predict(text)
        
        print(f"\nPredicted Emotion: {result['emotion'].upper()}")
        print(f"  Confidence: {result['confidence']:.2%}\n")


def demo_mode():
    predictor = EmotionPredictor()
    
    print("="*60)
    print("EMOTION RECOGNITION - DEMO")
    print("="*60 + "\n")
    
    examples = [
        "I'm so happy to see you today!",
        "This is absolutely terrible, I can't believe this happened.",
        "I don't know what to do, I'm completely confused.",
        "Everything is fine, nothing special.",
        "I am very angry at you!",
        "I miss you so much.",
    ]
    
    for i, text in enumerate(examples, 1):
        print(f"Example {i}: '{text}'")
        result = predictor.predict(text)
        print(f"  Emotion: {result['emotion'].upper()}")
        print(f"  Confidence: {result['confidence']:.2%}\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Emotion Recognition Inference')
    parser.add_argument('--mode', type=str, default='interactive', 
                       choices=['demo', 'interactive'],
                       help='Inference mode')
    parser.add_argument('--text', type=str, help='Single text to analyze')
    
    args = parser.parse_args()
    
    if args.text:
        predictor = EmotionPredictor()
        result = predictor.predict(args.text)
        print(f"Text: {args.text}")
        print(f"Emotion: {result['emotion'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")
    elif args.mode == 'demo':
        demo_mode()
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
