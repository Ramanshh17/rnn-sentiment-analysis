"""
LSTM Implementation for Sentiment Analysis
Based on: "Long Short-Term Memory" (Hochreiter & Schmidhuber, 1997)

This module implements:
1. LSTM cell with forget, input, and output gates
2. Bidirectional LSTM for better context understanding
3. Multi-layer LSTM for deep feature extraction
4. Sentiment classification head

Paper: https://www.bioinf.jku.at/publications/older/2604.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMSentiment(nn.Module):
    """
    LSTM Model for Sentiment Classification
    
    Architecture:
        Input (text indices) 
            ↓
        Embedding Layer (word → vectors)
            ↓
        Bidirectional LSTM (2 layers)
            ↓
        Fully Connected Layers
            ↓
        Output (sentiment logits)
    
    Args:
        vocab_size (int): Size of vocabulary
        embedding_dim (int): Dimension of word embeddings (default: 300)
        hidden_dim (int): Number of hidden units in LSTM (default: 256)
        output_dim (int): Number of output classes (default: 2 for binary)
        n_layers (int): Number of LSTM layers (default: 2)
        bidirectional (bool): Use bidirectional LSTM (default: True)
        dropout (float): Dropout rate (default: 0.5)
        pad_idx (int): Padding token index (default: 0)
    
    Example:
        >>> model = LSTMSentiment(vocab_size=10000)
        >>> text = torch.randint(0, 10000, (32, 100))  # batch=32, seq_len=100
        >>> lengths = torch.randint(50, 100, (32,))
        >>> output = model(text, lengths)
        >>> print(output.shape)  # torch.Size([32, 2])
    """
    
    def __init__(
        self, 
        vocab_size,
        embedding_dim=300,
        hidden_dim=256,
        output_dim=2,
        n_layers=2,
        bidirectional=True,
        dropout=0.5,
        pad_idx=0
    ):
        super(LSTMSentiment, self).__init__()
        
        # Store configuration
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        
        # Embedding layer
        # Converts word indices to dense vectors
        # vocab_size: number of unique words
        # embedding_dim: size of word vectors
        # padding_idx: ignore padding tokens during training
        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dim, 
            padding_idx=pad_idx
        )
        
        # LSTM layer
        # The core component from the paper
        # Processes sequences while maintaining memory
        self.lstm = nn.LSTM(
            input_size=embedding_dim,      # Input dimension
            hidden_size=hidden_dim,        # Hidden state dimension
            num_layers=n_layers,           # Stack multiple LSTMs
            bidirectional=bidirectional,   # Process forward & backward
            dropout=dropout if n_layers > 1 else 0,  # Dropout between layers
            batch_first=True               # Input shape: (batch, seq, feature)
        )
        
        # Fully connected layers for classification
        # If bidirectional, we concatenate forward and backward hidden states
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fc_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, text, text_lengths):
        """
        Forward pass through the network
        
        Args:
            text (torch.Tensor): Input text indices
                Shape: (batch_size, seq_len)
            text_lengths (torch.Tensor): Original lengths of sequences
                Shape: (batch_size,)
        
        Returns:
            torch.Tensor: Class logits
                Shape: (batch_size, output_dim)
        
        Process:
            1. Convert word indices to embeddings
            2. Pack sequences (for efficiency with variable lengths)
            3. Process through LSTM
            4. Extract final hidden state
            5. Pass through classification layers
        """
        
        # Step 1: Embedding
        # text shape: (batch_size, seq_len)
        embedded = self.embedding(text)
        # embedded shape: (batch_size, seq_len, embedding_dim)
        
        # Step 2: Pack padded sequences
        # This tells the LSTM to ignore padding tokens
        # Improves efficiency and accuracy
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, 
            text_lengths.cpu(),      # Lengths must be on CPU
            batch_first=True, 
            enforce_sorted=False     # Don't require sorted lengths
        )
        
        # Step 3: LSTM processing
        # The LSTM processes the sequence and returns:
        # - packed_output: outputs at each time step
        # - hidden: final hidden state
        # - cell: final cell state (long-term memory)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # Step 4: Extract final hidden state
        # For bidirectional LSTM:
        # hidden shape: (n_layers * 2, batch_size, hidden_dim)
        # We want the last layer's forward and backward states
        
        if self.bidirectional:
            # Get last layer's forward hidden state
            hidden_fwd = hidden[-2, :, :]  # Shape: (batch_size, hidden_dim)
            
            # Get last layer's backward hidden state
            hidden_bwd = hidden[-1, :, :]  # Shape: (batch_size, hidden_dim)
            
            # Concatenate forward and backward
            hidden = torch.cat((hidden_fwd, hidden_bwd), dim=1)
            # hidden shape: (batch_size, hidden_dim * 2)
        else:
            # For unidirectional, just take last layer
            hidden = hidden[-1, :, :]
            # hidden shape: (batch_size, hidden_dim)
        
        # Step 5: Classification
        # Pass through fully connected layers
        predictions = self.fc(hidden)
        # predictions shape: (batch_size, output_dim)
        
        return predictions


class SimpleLSTM(nn.Module):
    """
    Simplified LSTM for easier understanding
    Good for beginners or quick experiments
    
    Differences from LSTMSentiment:
    - Single LSTM layer (not 2)
    - Simpler classifier
    - Fewer configuration options
    """
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, output_dim=2):
        super(SimpleLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        
        # Use the final hidden state
        hidden = hidden[-1, :, :]
        hidden = self.dropout(hidden)
        
        output = self.fc(hidden)
        return output


class GRUSentiment(nn.Module):
    """
    GRU (Gated Recurrent Unit) alternative to LSTM
    
    Differences from LSTM:
    - Simpler architecture (2 gates instead of 3)
    - No separate cell state
    - Often performs similarly to LSTM
    - Faster training
    
    Use when:
    - You want faster training
    - LSTM is overfitting
    - Simpler model is preferred
    """
    
    def __init__(
        self, 
        vocab_size,
        embedding_dim=300,
        hidden_dim=256,
        output_dim=2,
        n_layers=2,
        bidirectional=True,
        dropout=0.5,
        pad_idx=0
    ):
        super(GRUSentiment, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        # GRU instead of LSTM
        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fc_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, text, text_lengths):
        embedded = self.embedding(text)
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # GRU returns only hidden state (no cell state)
        packed_output, hidden = self.gru(packed_embedded)
        
        if self.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        
        output = self.fc(hidden)
        return output


# Utility functions

def count_parameters(model):
    """
    Count the number of trainable parameters in a model
    
    Args:
        model (nn.Module): PyTorch model
    
    Returns:
        int: Number of trainable parameters
    
    Example:
        >>> model = LSTMSentiment(vocab_size=10000)
        >>> params = count_parameters(model)
        >>> print(f"Model has {params:,} trainable parameters")
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_weights(model):
    """
    Initialize model weights with Xavier uniform initialization
    
    Args:
        model (nn.Module): PyTorch model
    
    Example:
        >>> model = LSTMSentiment(vocab_size=10000)
        >>> init_weights(model)
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param.data)
        elif 'bias' in name:
            nn.init.constant_(param.data, 0)


# Testing code (runs when you execute: python src/models/lstm.py)
if __name__ == "__main__":
    print("="*70)
    print("Testing LSTM Model")
    print("="*70)
    
    # Test parameters
    vocab_size = 5000
    batch_size = 4
    seq_len = 10
    
    # Create model
    print("\n1. Creating LSTMSentiment model...")
    model = LSTMSentiment(vocab_size=vocab_size)
    print(f"   ✓ Model created successfully!")
    
    # Count parameters
    num_params = count_parameters(model)
    print(f"\n2. Model Statistics:")
    print(f"   Total parameters: {num_params:,}")
    
    # Print architecture
    print(f"\n3. Model Architecture:")
    print(model)
    
    # Test forward pass
    print(f"\n4. Testing forward pass...")
    
    # Create dummy input
    text = torch.randint(0, vocab_size, (batch_size, seq_len))
    text_lengths = torch.tensor([10, 8, 6, 5])
    
    print(f"   Input text shape: {text.shape}")
    print(f"   Sequence lengths: {text_lengths.tolist()}")
    
    # Forward pass
    output = model(text, text_lengths)
    
    print(f"   Output shape: {output.shape}")
    print(f"   Expected shape: ({batch_size}, 2)")
    
    # Verify output
    assert output.shape == (batch_size, 2), "Output shape mismatch!"
    print(f"\n✓ All tests passed!")
    
    # Test other models
    print(f"\n5. Testing SimpleLSTM...")
    simple_model = SimpleLSTM(vocab_size=vocab_size)
    simple_output = simple_model(text, text_lengths)
    print(f"   ✓ SimpleLSTM works! Output shape: {simple_output.shape}")
    
    print(f"\n6. Testing GRUSentiment...")
    gru_model = GRUSentiment(vocab_size=vocab_size)
    gru_output = gru_model(text, text_lengths)
    print(f"   ✓ GRUSentiment works! Output shape: {gru_output.shape}")
    
    print("\n" + "="*70)
    print("✓ All models working correctly!")
    print("="*70 + "\n")