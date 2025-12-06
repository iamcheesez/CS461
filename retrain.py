"""
Retrain script using feedback data
This script retrains the model using images collected from user feedback
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from datetime import datetime
import argparse


class FeedbackDataset(Dataset):
    """Dataset for feedback images"""
    
    def __init__(self, feedback_dir: str, class_mapping_path: str, transform=None):
        self.feedback_dir = Path(feedback_dir) / 'images'
        self.transform = transform
        
        # Load class mapping
        with open(class_mapping_path, 'r') as f:
            self.class_to_idx = json.load(f)
        
        self.num_classes = len(self.class_to_idx)
        
        # Collect all feedback images
        self.samples = []
        if self.feedback_dir.exists():
            for class_folder in self.feedback_dir.iterdir():
                if class_folder.is_dir() and class_folder.name in self.class_to_idx:
                    class_idx = self.class_to_idx[class_folder.name]
                    for img_path in class_folder.glob('*'):
                        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                            self.samples.append((str(img_path), class_idx))
        
        print(f"Found {len(self.samples)} feedback images across {len(set(s[1] for s in self.samples))} classes")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def load_model(model_path: str, num_classes: int, device: torch.device):
    """Load existing model for fine-tuning"""
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    return model, checkpoint


def fine_tune(
    model_path: str = 'models/best_model.pth',
    class_mapping_path: str = 'models/class_mapping.json',
    feedback_dir: str = 'feedback_data',
    output_path: str = None,
    epochs: int = 5,
    batch_size: int = 8,
    learning_rate: float = 0.0001,
    device: str = 'cuda'
):
    """Fine-tune the model using feedback data"""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load class mapping
    with open(class_mapping_path, 'r') as f:
        class_to_idx = json.load(f)
    num_classes = len(class_to_idx)
    
    # Define transforms (with augmentation)
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = FeedbackDataset(feedback_dir, class_mapping_path, transform=train_transform)
    
    if len(dataset) == 0:
        print("No feedback data found. Nothing to train on.")
        return False
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Load model
    print(f"Loading model from {model_path}...")
    model, checkpoint = load_model(model_path, num_classes, device)
    
    # Freeze most layers, only train fc layer and last few layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze fc layer
    for param in model.fc.parameters():
        param.requires_grad = True
    
    # Unfreeze last conv block (layer4)
    for param in model.layer4.parameters():
        param.requires_grad = True
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    # Training loop
    print(f"\nStarting fine-tuning for {epochs} epochs...")
    print(f"Training on {len(dataset)} feedback samples")
    print("-" * 50)
    
    model.train()
    best_loss = float('inf')
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")
        
        if epoch_loss < best_loss:
            best_loss = epoch_loss
    
    # Save fine-tuned model
    if output_path is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f'models/finetuned_model_{timestamp}.pth'
    
    # Update checkpoint
    checkpoint['model_state_dict'] = model.state_dict()
    checkpoint['finetuned_on'] = datetime.now().isoformat()
    checkpoint['feedback_samples'] = len(dataset)
    
    torch.save(checkpoint, output_path)
    print(f"\nFine-tuned model saved to {output_path}")
    
    # Also update best_model.pth
    torch.save(checkpoint, model_path)
    print(f"Updated {model_path}")
    
    return True


def get_feedback_stats(feedback_dir: str = 'feedback_data'):
    """Get statistics about collected feedback"""
    log_path = os.path.join(feedback_dir, 'feedback_log.json')
    images_dir = os.path.join(feedback_dir, 'images')
    
    stats = {
        'total_feedback': 0,
        'correct': 0,
        'incorrect': 0,
        'saved_images': 0,
        'classes_with_data': []
    }
    
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            logs = json.load(f)
        
        stats['total_feedback'] = len(logs)
        stats['correct'] = sum(1 for log in logs if log.get('is_correct'))
        stats['incorrect'] = stats['total_feedback'] - stats['correct']
        stats['saved_images'] = sum(1 for log in logs if log.get('saved_for_training'))
    
    if os.path.exists(images_dir):
        for class_folder in os.listdir(images_dir):
            class_path = os.path.join(images_dir, class_folder)
            if os.path.isdir(class_path):
                count = len([f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
                if count > 0:
                    stats['classes_with_data'].append({'class': class_folder, 'count': count})
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Fine-tune model with feedback data')
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--classes', type=str, default='models/class_mapping.json',
                       help='Path to class mapping JSON')
    parser.add_argument('--feedback', type=str, default='feedback_data',
                       help='Path to feedback data directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for fine-tuned model')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--stats', action='store_true',
                       help='Show feedback statistics only')
    
    args = parser.parse_args()
    
    if args.stats:
        print("\n" + "=" * 50)
        print("Feedback Statistics")
        print("=" * 50)
        
        stats = get_feedback_stats(args.feedback)
        print(f"Total feedback received: {stats['total_feedback']}")
        print(f"  - Correct predictions: {stats['correct']}")
        print(f"  - Incorrect predictions: {stats['incorrect']}")
        print(f"  - Images saved for training: {stats['saved_images']}")
        
        if stats['classes_with_data']:
            print(f"\nClasses with feedback data ({len(stats['classes_with_data'])}):")
            for item in sorted(stats['classes_with_data'], key=lambda x: x['count'], reverse=True)[:10]:
                print(f"  - {item['class']}: {item['count']} images")
        
        return
    
    # Run fine-tuning
    fine_tune(
        model_path=args.model,
        class_mapping_path=args.classes,
        feedback_dir=args.feedback,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device
    )


if __name__ == "__main__":
    main()
