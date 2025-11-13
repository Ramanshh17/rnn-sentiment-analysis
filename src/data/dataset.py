"""
Data Loading and Preprocessing for IMDB Dataset

This module handles:
1. Downloading IMDB dataset
2. Building vocabulary
3. Text preprocessing
4. Creating DataLoaders
"""

import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import re
import os
import urllib.request
import tarfile
from tqdm import tqdm


class IMDBDataset(Dataset):
    """
    IMDB Movie Review Dataset
    
    50,000 reviews labeled as positive (1) or negative (0)
    """
    
    def __init__(self, texts, labels, vocab, max_len=256):
        """
        Args:
            texts: List of review texts
            labels: List of labels (0 or 1)
            vocab: Vocabulary dictionary
            max_len: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def text_to_sequence(self, text):
        """
        Convert text to sequence of indices
        
        Steps:
        1. Tokenize (split into words)
        2. Convert to lowercase
        3. Map to vocabulary indices
        4. Pad/truncate to max_len
        """
        # Simple tokenization
        tokens = self.tokenize(text)
        
        # Convert to indices
        sequence = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        # Get original length before padding
        length = min(len(sequence), self.max_len)
        
        # Pad or truncate
        if len(sequence) < self.max_len:
            sequence = sequence + [self.vocab['<PAD>']] * (self.max_len - len(sequence))
        else:
            sequence = sequence[:self.max_len]
        
        return sequence, length
    
    def tokenize(self, text):
        """Simple word tokenization"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Keep only alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s]', '', text)
        
        # Split into words
        tokens = text.split()
        
        return tokens
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        sequence, length = self.text_to_sequence(text)
        
        return {
            'text': torch.tensor(sequence, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'length': torch.tensor(length, dtype=torch.long)
        }


def build_vocab(texts, max_vocab_size=10000, min_freq=2):
    """
    Build vocabulary from texts
    
    Args:
        texts: List of text strings
        max_vocab_size: Maximum vocabulary size
        min_freq: Minimum frequency for a word to be included
        
    Returns:
        vocab: Dictionary mapping words to indices
    """
    print("Building vocabulary...")
    
    # Count all words
    counter = Counter()
    for text in tqdm(texts):
        tokens = text.lower().split()
        counter.update(tokens)
    
    # Create vocabulary
    # Reserve indices for special tokens
    vocab = {
        '<PAD>': 0,  # Padding
        '<UNK>': 1,  # Unknown words
    }
    
    # Add most common words
    for word, freq in counter.most_common(max_vocab_size - 2):
        if freq >= min_freq:
            vocab[word] = len(vocab)
    
    print(f"Vocabulary size: {len(vocab)}")
    return vocab


def load_imdb_data(data_dir='./data/imdb'):
    """
    Load IMDB dataset
    
    For now, we'll create a synthetic dataset.
    Replace this with actual IMDB data loading.
    """
    print("Loading IMDB dataset...")
    
    # Check if data exists
    if not os.path.exists(data_dir):
        print(f"Creating synthetic dataset for demonstration...")
        print("To use real IMDB data, download from: http://ai.stanford.edu/~amaas/data/sentiment/")
        
        # Create synthetic data for demonstration
        train_texts, train_labels = create_synthetic_data(1000)
        test_texts, test_labels = create_synthetic_data(200)
    else:
        # Load real IMDB data
        train_texts, train_labels = load_real_imdb(data_dir, 'train')
        test_texts, test_labels = load_real_imdb(data_dir, 'test')
    
    return train_texts, train_labels, test_texts, test_labels


def create_synthetic_data(num_samples):
    """
    Create synthetic movie review data for testing
    Replace with real data for actual use
    """
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 
                      'fantastic', 'love', 'best', 'perfect', 'brilliant']
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 
                      'hate', 'disappointing', 'poor', 'waste', 'boring']
    neutral_words = ['movie', 'film', 'watch', 'story', 'acting', 'plot', 
                     'director', 'character', 'scene', 'performance']
    
    texts = []
    labels = []
    
    for i in range(num_samples):
        # Random label
        label = i % 2
        
        # Create review
        if label == 1:  # Positive
            words = positive_words + neutral_words
        else:  # Negative
            words = negative_words + neutral_words
        
        # Random review length
        import random
        review_length = random.randint(20, 100)
        review = ' '.join(random.choices(words, k=review_length))
        
        texts.append(review)
        labels.append(label)
    
    return texts, labels


def load_real_imdb(data_dir, split='train'):
    """
    Load real IMDB data from disk
    
    Expected structure:
    data_dir/
        train/
            pos/
            neg/
        test/
            pos/
            neg/
    """
    texts = []
    labels = []
    
    # Load positive reviews
    pos_dir = os.path.join(data_dir, split, 'pos')
    if os.path.exists(pos_dir):
        for filename in os.listdir(pos_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(pos_dir, filename), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                    labels.append(1)
    
    # Load negative reviews
    neg_dir = os.path.join(data_dir, split, 'neg')
    if os.path.exists(neg_dir):
        for filename in os.listdir(neg_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(neg_dir, filename), 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                    labels.append(0)
    
    return texts, labels


def get_data_loaders(batch_size=32, max_vocab_size=10000, max_len=256):
    """
    Create train and test data loaders
    
    Returns:
        train_loader, test_loader, vocab
    """
    # Load data
    train_texts, train_labels, test_texts, test_labels = load_imdb_data()
    
    # Build vocabulary from training data
    vocab = build_vocab(train_texts, max_vocab_size=max_vocab_size)
    
    # Create datasets
    train_dataset = IMDBDataset(train_texts, train_labels, vocab, max_len)
    test_dataset = IMDBDataset(test_texts, test_labels, vocab, max_len)
    
    print(f"\nDataset Statistics:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Max sequence length: {max_len}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, test_loader, vocab


# Test the data loading
if __name__ == "__main__":
    print("Testing data loading...\n")
    
    train_loader, test_loader, vocab = get_data_loaders(batch_size=4)
    
    # Get one batch
    batch = next(iter(train_loader))
    
    print(f"\nBatch contents:")
    print(f"  Text shape: {batch['text'].shape}")
    print(f"  Label shape: {batch['label'].shape}")
    print(f"  Length shape: {batch['length'].shape}")
    
    print(f"\n✓ Data loading works correctly!")