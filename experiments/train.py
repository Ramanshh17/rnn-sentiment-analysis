"""
Main Training Script for LSTM Sentiment Analysis
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Fix Python path
if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

# Now import
from src.models.lstm import LSTMSentiment, count_parameters  # ✓ CORRECT
from src.data.dataset import get_data_loaders
from src.utils.trainer import Trainer


def main():
    """Main training function"""
    
    # Configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    BATCH_SIZE = 64
    MAX_VOCAB_SIZE = 10000
    MAX_SEQ_LEN = 256
    
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 256
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    OUTPUT_DIM = 2
    
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    
    CHECKPOINT_DIR = 'checkpoints'
    LOG_DIR = 'runs/lstm_sentiment'
    
    # Print configuration
    print("\n" + "="*70)
    print("LSTM SENTIMENT ANALYSIS - TRAINING CONFIGURATION")
    print("="*70)
    print(f"\n📱 Device: {DEVICE}")
    
    print(f"\n📊 Data Configuration:")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Max Vocab Size: {MAX_VOCAB_SIZE}")
    print(f"   Max Sequence Length: {MAX_SEQ_LEN}")
    
    print(f"\n🏗️  Model Configuration:")
    print(f"   Embedding Dim: {EMBEDDING_DIM}")
    print(f"   Hidden Dim: {HIDDEN_DIM}")
    print(f"   Number of Layers: {N_LAYERS}")
    print(f"   Bidirectional: {BIDIRECTIONAL}")
    print(f"   Dropout: {DROPOUT}")
    
    print(f"\n🎓 Training Configuration:")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Number of Epochs: {NUM_EPOCHS}")
    
    print("\n" + "="*70 + "\n")
    
    # Load data
    print("📁 Loading dataset...")
    print("-" * 70)
    
    train_loader, test_loader, vocab = get_data_loaders(
        batch_size=BATCH_SIZE,
        max_vocab_size=MAX_VOCAB_SIZE,
        max_len=MAX_SEQ_LEN
    )
    
    vocab_size = len(vocab)
    
    print(f"\n✓ Data loaded successfully!")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Create model
    print(f"\n🏗️  Creating LSTM model...")
    print("-" * 70)
    
    model = LSTMSentiment(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        n_layers=N_LAYERS,
        bidirectional=BIDIRECTIONAL,
        dropout=DROPOUT,
        pad_idx=vocab['<PAD>']
    )
    
    num_params = count_parameters(model)
    print(f"\n✓ Model created!")
    print(f"  Total parameters: {num_params:,}")
    
    # Setup training
    print(f"\n⚙️  Setting up training...")
    print("-" * 70)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"✓ Training setup complete!")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=DEVICE,
        checkpoint_dir=CHECKPOINT_DIR,
        log_dir=LOG_DIR
    )
    
    # Train model
    try:
        trainer.train(num_epochs=NUM_EPOCHS)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Saving current model...")
        trainer.save_checkpoint('interrupted_model.pth')
        print("✓ Model saved!")
    
    # Training complete
    print(f"\n🎉 Training session complete!")
    print(f"\n📊 To view training progress:")
    print(f"   tensorboard --logdir={LOG_DIR}")
    print(f"\n💾 Model saved in:")
    print(f"   {CHECKPOINT_DIR}\\best_model.pth")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()