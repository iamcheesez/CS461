"""
Train ResNet50 model for dog breed classification
"""
import os
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import create_data_loaders


class DogBreedClassifier:
    """ResNet50-based dog breed classifier"""
    
    def __init__(self, num_classes: int, device: str = 'cuda'):
        """
        Args:
            num_classes: Number of dog breeds to classify
            device: Device to run training on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load pre-trained ResNet50
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Modify final layer for our number of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
    
    def train_epoch(self, train_loader, criterion, optimizer) -> tuple:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, criterion) -> tuple:
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct / total:.2f}%'
                })
        
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, num_epochs: int = 20, 
              learning_rate: float = 0.001, save_dir: str = 'models'):
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            learning_rate: Initial learning rate
            save_dir: Directory to save model checkpoints
        """
        # Create save directory
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3
        )
        
        best_val_acc = 0.0
        start_time = time.time()
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 40)
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }
                torch.save(checkpoint, save_path / 'best_model.pth')
                print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
            
            # Save latest model
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }
            torch.save(checkpoint, save_path / 'latest_model.pth')
        
        # Training complete
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"Training completed in {elapsed_time / 60:.2f} minutes")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        
        # Save training history
        with open(save_path / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=4)
        
        # Plot training curves
        self.plot_training_curves(save_path / 'training_curves.png')
    
    def plot_training_curves(self, save_path: str):
        """Plot and save training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(self.history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(self.history['val_loss'], label='Val Loss', marker='o')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(self.history['train_acc'], label='Train Accuracy', marker='o')
        axes[1].plot(self.history['val_acc'], label='Val Accuracy', marker='o')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Training curves saved to {save_path}")


def main():
    """Main training function"""
    # Configuration
    DATA_DIR = "images/Images"
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    NUM_WORKERS = 4
    
    print("Dog Breed Classification with ResNet50")
    print("=" * 60)
    
    # Create data loaders
    print("\nLoading dataset...")
    train_loader, val_loader, class_mapping = create_data_loaders(
        DATA_DIR, 
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )
    
    num_classes = len(class_mapping)
    print(f"Number of dog breeds: {num_classes}")
    
    # Save class mapping
    save_dir = Path('models')
    save_dir.mkdir(exist_ok=True)
    with open(save_dir / 'class_mapping.json', 'w') as f:
        json.dump(class_mapping, f, indent=4)
    print(f"Class mapping saved to {save_dir / 'class_mapping.json'}")
    
    # Create and train model
    classifier = DogBreedClassifier(num_classes=num_classes)
    classifier.train(
        train_loader, 
        val_loader, 
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE
    )
    
    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
