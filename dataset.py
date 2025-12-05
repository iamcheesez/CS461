"""
Dataset preparation for dog breed classification
"""
import os
from pathlib import Path
from typing import Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random


class DogBreedDataset(Dataset):
    """Custom dataset for dog breed classification from ImageNet structure"""
    
    def __init__(self, root_dir: str, transform=None, train: bool = True, train_split: float = 0.8):
        """
        Args:
            root_dir: Path to Images folder containing breed subfolders
            transform: Optional transform to be applied on images
            train: If True, use training split; if False, use validation split
            train_split: Fraction of data to use for training
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.train = train
        
        # Get all breed folders and create class mapping
        self.breed_folders = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {folder.name: idx for idx, folder in enumerate(self.breed_folders)}
        self.idx_to_class = {idx: breed_name for breed_name, idx in self.class_to_idx.items()}
        
        # Collect all image paths and labels
        self.samples = []
        for breed_folder in self.breed_folders:
            breed_name = breed_folder.name
            class_idx = self.class_to_idx[breed_name]
            
            # Get all images in this breed folder
            image_files = sorted(list(breed_folder.glob("*.jpg")) + list(breed_folder.glob("*.JPEG")))
            
            for img_path in image_files:
                self.samples.append((str(img_path), class_idx))
        
        # Shuffle and split into train/val
        random.seed(42)
        random.shuffle(self.samples)
        
        split_idx = int(len(self.samples) * train_split)
        if self.train:
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]
        
        print(f"{'Training' if train else 'Validation'} dataset: {len(self.samples)} images from {len(self.breed_folders)} breeds")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a black image if loading fails
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_breed_name(self, class_idx: int) -> str:
        """Get human-readable breed name from class index"""
        folder_name = self.idx_to_class[class_idx]
        # Extract breed name from folder format "n02085620-Chihuahua"
        breed_name = folder_name.split('-', 1)[1] if '-' in folder_name else folder_name
        return breed_name.replace('_', ' ')


def get_transforms(train: bool = True) -> transforms.Compose:
    """Get data transforms for training or validation"""
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def create_data_loaders(data_dir: str, batch_size: int = 32, num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders"""
    
    train_dataset = DogBreedDataset(
        root_dir=data_dir,
        transform=get_transforms(train=True),
        train=True
    )
    
    val_dataset = DogBreedDataset(
        root_dir=data_dir,
        transform=get_transforms(train=False),
        train=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_dataset.class_to_idx


if __name__ == "__main__":
    # Test the dataset
    images_dir = "images/Images"
    train_loader, val_loader, class_mapping = create_data_loaders(images_dir, batch_size=8)
    
    print(f"\nNumber of classes: {len(class_mapping)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Test loading a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Sample labels: {labels[:5]}")
