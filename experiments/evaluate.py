"""
Evaluate trained LSTM model
"""

import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.lstm import LSTMSentiment
from src.data.dataset import get_data_loaders
from tqdm import tqdm


def evaluate_model(model_path='checkpoints/best_model.pth'):
    """Evaluate saved model"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load data
    print("Loading data...")
    _, test_loader, vocab = get_data_loaders(batch_size=32)
    
    # Create model
    model = LSTMSentiment(
        vocab_size=len(vocab),
        embedding_dim=300,
        hidden_dim=256,
        n_layers=2,
        bidirectional=True
    ).to(device)
    
    # Load checkpoint
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Evaluate
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    print("\nEvaluating...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            text = batch['text'].to(device)
            labels = batch['label'].to(device)
            lengths = batch['length']
            
            predictions = model(text, lengths)
            pred_classes = predictions.argmax(dim=1)
            
            correct += (pred_classes == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(pred_classes.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    
    print(f"\n{'='*50}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Correct: {correct}/{total}")
    print(f"{'='*50}\n")
    
    return all_preds, all_labels


if __name__ == '__main__':
    evaluate_model()