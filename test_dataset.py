"""
Quick test script to verify dataset loading
"""
from dataset import create_data_loaders
from pathlib import Path

def test_dataset():
    """Test if the dataset can be loaded properly"""
    print("Testing Dog Breed Dataset")
    print("=" * 60)
    
    # Check if images directory exists
    images_dir = Path("images/Images")
    if not images_dir.exists():
        print(f"❌ Error: Images directory not found at {images_dir}")
        print("   Please ensure the 'images/Images' folder exists in the current directory")
        return False
    
    print(f"✓ Found images directory: {images_dir}")
    
    # Count breed folders
    breed_folders = [d for d in images_dir.iterdir() if d.is_dir()]
    print(f"✓ Found {len(breed_folders)} breed folders")
    
    # Try to create data loaders
    try:
        print("\nCreating data loaders...")
        train_loader, val_loader, class_mapping = create_data_loaders(
            str(images_dir),
            batch_size=8,
            num_workers=0  # Use 0 for testing
        )
        
        print(f"✓ Data loaders created successfully")
        print(f"  - Number of classes: {len(class_mapping)}")
        print(f"  - Training batches: {len(train_loader)}")
        print(f"  - Validation batches: {len(val_loader)}")
        
        # Try to load one batch
        print("\nLoading a test batch...")
        images, labels = next(iter(train_loader))
        print(f"✓ Successfully loaded batch")
        print(f"  - Batch shape: {images.shape}")
        print(f"  - Labels shape: {labels.shape}")
        print(f"  - Sample labels: {labels[:5].tolist()}")
        
        # Show some breed names
        print("\nSample breed names:")
        sample_breeds = list(class_mapping.keys())[:10]
        for breed in sample_breeds:
            breed_name = breed.split('-', 1)[1] if '-' in breed else breed
            print(f"  - {breed_name.replace('_', ' ')}")
        
        print("\n" + "=" * 60)
        print("✓ Dataset test PASSED!")
        print("You can now run: python train.py")
        return True
        
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_dataset()
