from transformers import pipeline
import sys
from pathlib import Path

class EmotionPredictor:
    
    def __init__(self, model_name="j-hartmann/emotion-english-distilroberta-base"):
        
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            top_k=None
        )
        
        print("Model loaded successfully!\n")
    
    def predict(self, text):
        results = self.classifier(text)[0]
        results = sorted(results, key=lambda x: x['score'], reverse=True)
        top_prediction = results[0]
        
        return {
            'predicted_emotion': top_prediction['label'],
            'confidence': top_prediction['score'],
            'all_scores': results
        }


def interactive_mode():
    
    print("="*60)
    print("EMOTION RECOGNITION")
    print("="*60)
    print("Emotions: anger, disgust, fear, joy, neutral, sadness, surprise")
    print("="*60 + "\n")
    
    predictor = EmotionPredictor()
    
    print("Enter text to analyze emotion (or 'quit' to exit)\n")
    
    while True:
        text = input("Enter text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not text:
            continue
        
        result = predictor.predict(text)
        
        print(f"\nPredicted Emotion: {result['predicted_emotion'].upper()}")
        print(f"  Confidence: {result['confidence']:.2%}")
        



def demo_predictions():
    
    print("="*60)
    print("EMOTION RECOGNITION - DEMO")
    print("="*60 + "\n")
    
    predictor = EmotionPredictor()
    
    examples = [
        "I'm so happy to see you today!",
        "This is absolutely terrible, I can't believe this happened.",
        "I don't know what to do, I'm completely confused.",
        "Everything is fine, nothing special.",
        "I am very angry at you!",
        "I miss you so much.",
        "This is disgusting!",
    ]
    
    for i, text in enumerate(examples, 1):
        print(f"Example {i}:")
        print(f"Text: '{text}'")
        
        result = predictor.predict(text)
        
        print(f"Predicted Emotion: {result['predicted_emotion'].upper()}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nAll Emotions:")
        for score in result['all_scores']:
            bar_length = int(score['score'] * 40)
            bar = "█" * bar_length
            print(f"  {score['label']:10s}: {bar} {score['score']:.2%}")
        print("\n" + "-"*60 + "\n")


def single_prediction(text):
    
    predictor = EmotionPredictor()
    result = predictor.predict(text)
    
    print(f"\nText: {text}")
    print(f"Emotion: {result['predicted_emotion'].upper()}")
    print(f"Confidence: {result['confidence']:.2%}\n")
    
    print("All Emotions:")
    for score in result['all_scores']:
        print(f"  {score['label']:10s}: {score['score']:.2%}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Emotion Recognition')
    parser.add_argument('--mode', type=str, default='interactive', 
                       choices=['demo', 'interactive'],
                       help='Inference mode: demo or interactive')
    parser.add_argument('--text', type=str, help='Text to analyze (for single prediction)')
    parser.add_argument('--model', type=str, 
                       default='j-hartmann/emotion-english-distilroberta-base',
                       help='Model name')
    
    args = parser.parse_args()
    
    if args.text:
        single_prediction(args.text)
    
    elif args.mode == 'demo':
        demo_predictions()
    
    elif args.mode == 'interactive':
        interactive_mode()


if __name__ == "__main__":
    main()
