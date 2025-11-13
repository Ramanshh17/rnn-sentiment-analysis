"""
Make predictions on new text
"""

import torch
import sys
import os
import re

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.lstm import LSTMSentiment
from src.data.dataset import get_data_loaders


def load_model_and_vocab():
    """Load trained model and vocabulary"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load vocab
    _, _, vocab = get_data_loaders(batch_size=1)
    
    # Create model
    model = LSTMSentiment(
        vocab_size=len(vocab),
        embedding_dim=300,
        hidden_dim=256,
        n_layers=2,
        bidirectional=True
    ).to(device)
    
    # Load weights
    checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, vocab, device


def preprocess_text(text, vocab, max_len=256):
    """Preprocess text for prediction"""
    # Tokenize
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    
    # Convert to indices
    sequence = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    length = min(len(sequence), max_len)
    
    # Pad
    if len(sequence) < max_len:
        sequence = sequence + [vocab['<PAD>']] * (max_len - len(sequence))
    else:
        sequence = sequence[:max_len]
    
    return torch.tensor([sequence]), torch.tensor([length])


def predict(text, model, vocab, device):
    """Predict sentiment of text"""
    # Preprocess
    text_tensor, length = preprocess_text(text, vocab)
    text_tensor = text_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        output = model(text_tensor, length)
        probabilities = torch.softmax(output, dim=1)
        prediction = output.argmax(dim=1).item()
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = probabilities[0][prediction].item() * 100
    
    return sentiment, confidence


def main():
    """Interactive prediction"""
    print("Loading model...")
    model, vocab, device = load_model_and_vocab()
    print("✓ Model loaded!\n")
    
    print("="*60)
    print("SENTIMENT ANALYSIS - Interactive Mode")
    print("="*60)
    print("Enter movie reviews to analyze sentiment")
    print("Type 'quit' to exit\n")
    
    while True:
        text = input("Review: ")
        
        if text.lower() == 'quit':
            break
        
        if not text.strip():
            continue
        
        sentiment, confidence = predict(text, model, vocab, device)
        
        print(f"\nSentiment: {sentiment}")
        print(f"Confidence: {confidence:.2f}%")
        print("-" * 60 + "\n")


if __name__ == '__main__':
    main()