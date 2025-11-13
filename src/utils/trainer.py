"""
Training and Evaluation Utilities
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
import time


class Trainer:
    """
    Trainer class for LSTM sentiment analysis
    
    Handles:
    - Training loop
    - Validation
    - Checkpointing
    - Logging to TensorBoard
    """
    
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        optimizer,
        criterion,
        device='cuda',
        checkpoint_dir='checkpoints',
        log_dir='runs'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_accuracy = 0.0
        self.train_losses = []
        self.test_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        
        epoch_loss = 0
        epoch_acc = 0
        total_samples = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1} [Train]')
        
        for batch in pbar:
            # Get data
            text = batch['text'].to(self.device)
            labels = batch['label'].to(self.device)
            lengths = batch['length']
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(text, lengths)
            
            # Calculate loss
            loss = self.criterion(predictions, labels)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients (important for RNNs!)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Calculate accuracy
            pred_classes = predictions.argmax(dim=1)
            correct = (pred_classes == labels).float().sum()
            acc = correct / labels.size(0)
            
            # Update statistics
            epoch_loss += loss.item() * labels.size(0)
            epoch_acc += correct.item()
            total_samples += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc.item() * 100:.2f}%'
            })
        
        # Calculate epoch metrics
        epoch_loss = epoch_loss / total_samples
        epoch_acc = epoch_acc / total_samples
        
        return epoch_loss, epoch_acc
    
    def evaluate(self):
        """Evaluate on test set"""
        self.model.eval()
        
        epoch_loss = 0
        epoch_acc = 0
        total_samples = 0
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc=f'Epoch {self.current_epoch + 1} [Test]')
            
            for batch in pbar:
                # Get data
                text = batch['text'].to(self.device)
                labels = batch['label'].to(self.device)
                lengths = batch['length']
                
                # Forward pass
                predictions = self.model(text, lengths)
                
                # Calculate loss
                loss = self.criterion(predictions, labels)
                
                # Calculate accuracy
                pred_classes = predictions.argmax(dim=1)
                correct = (pred_classes == labels).float().sum()
                acc = correct / labels.size(0)
                
                # Update statistics
                epoch_loss += loss.item() * labels.size(0)
                epoch_acc += correct.item()
                total_samples += labels.size(0)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{acc.item() * 100:.2f}%'
                })
        
        # Calculate epoch metrics
        epoch_loss = epoch_loss / total_samples
        epoch_acc = epoch_acc / total_samples
        
        return epoch_loss, epoch_acc
    
    def train(self, num_epochs):
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs to train
        """
        print(f"\n{'='*70}")
        print(f"{'Starting Training':^70}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Evaluate
            test_loss, test_acc = self.evaluate()
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_acc)
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/test', test_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc * 100, epoch)
            self.writer.add_scalar('Accuracy/test', test_acc * 100, epoch)
            
            # Print epoch summary
            print(f"\n{'─'*70}")
            print(f"Epoch {epoch + 1}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc * 100:.2f}%")
            print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc * 100:.2f}%")
            
            # Save best model
            if test_acc > self.best_accuracy:
                self.best_accuracy = test_acc
                self.save_checkpoint('best_model.pth')
                print(f"  ✓ New best model saved! (Accuracy: {test_acc * 100:.2f}%)")
            
            print(f"{'─'*70}\n")
        
        # Training complete
        total_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"{'Training Complete!':^70}")
        print(f"{'='*70}")
        print(f"  Total Time: {total_time / 60:.2f} minutes")
        print(f"  Best Test Accuracy: {self.best_accuracy * 100:.2f}%")
        print(f"  Model saved in: {self.checkpoint_dir}/best_model.pth")
        print(f"{'='*70}\n")
        
        self.writer.close()
    
    def save_checkpoint(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_accuracy': self.best_accuracy,
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'train_accuracies': self.train_accuracies,
            'test_accuracies': self.test_accuracies,
        }
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_accuracy = checkpoint['best_accuracy']
        self.train_losses = checkpoint['train_losses']
        self.test_losses = checkpoint['test_losses']
        self.train_accuracies = checkpoint['train_accuracies']
        self.test_accuracies = checkpoint['test_accuracies']
        
        print(f"✓ Loaded checkpoint from epoch {self.current_epoch}")
        print(f"  Best accuracy: {self.best_accuracy * 100:.2f}%")


def binary_accuracy(preds, y):
    """
    Calculate binary classification accuracy
    
    Args:
        preds: Predictions (batch_size, 2)
        y: Labels (batch_size)
    
    Returns:
        accuracy: Float between 0 and 1
    """
    pred_classes = preds.argmax(dim=1)
    correct = (pred_classes == y).float()
    acc = correct.sum() / len(correct)
    return acc