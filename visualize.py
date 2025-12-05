"""
Visualize model predictions on sample images
"""
import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import json
from pathlib import Path
import random
from predict import DogBreedPredictor


def visualize_predictions(image_paths, predictor, save_path=None):
    """
    Visualize predictions for multiple images
    
    Args:
        image_paths: List of image paths to visualize
        predictor: DogBreedPredictor instance
        save_path: Optional path to save the visualization
    """
    n_images = len(image_paths)
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if n_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, img_path in enumerate(image_paths):
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Get prediction
        predictions = predictor.predict(img_path, top_k=3)
        
        # Display image
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        # Create title with predictions
        if predictions:
            title = f"Top 3 Predictions:\n"
            for i, (breed, prob) in enumerate(predictions, 1):
                title += f"{i}. {breed}: {prob*100:.1f}%\n"
        else:
            title = "No predictions"
        
        axes[idx].set_title(title, fontsize=10, pad=10)
    
    # Hide empty subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


def test_on_random_samples(images_dir, predictor, num_samples=8):
    """Test model on random samples from the dataset"""
    images_dir = Path(images_dir)
    
    # Get all image files
    all_images = []
    for breed_folder in images_dir.iterdir():
        if breed_folder.is_dir():
            all_images.extend(list(breed_folder.glob("*.jpg")))
    
    # Select random samples
    if len(all_images) < num_samples:
        num_samples = len(all_images)
    
    sample_images = random.sample(all_images, num_samples)
    
    print(f"\nTesting on {num_samples} random images from dataset...")
    visualize_predictions(sample_images, predictor, save_path="sample_predictions.png")


def main():
    """Main visualization function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize Dog Breed Predictions')
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--classes', type=str, default='models/class_mapping.json',
                       help='Path to class mapping')
    parser.add_argument('--images', type=str, nargs='+',
                       help='Specific image paths to visualize')
    parser.add_argument('--random', action='store_true',
                       help='Test on random samples from dataset')
    parser.add_argument('--num-samples', type=int, default=8,
                       help='Number of random samples to test')
    parser.add_argument('--data-dir', type=str, default='images/Images',
                       help='Path to images directory for random sampling')
    
    args = parser.parse_args()
    
    # Create predictor
    print("Loading model...")
    predictor = DogBreedPredictor(
        model_path=args.model,
        class_mapping_path=args.classes
    )
    
    if args.random:
        # Test on random samples
        test_on_random_samples(args.data_dir, predictor, args.num_samples)
    elif args.images:
        # Visualize specific images
        visualize_predictions(args.images, predictor, save_path="predictions.png")
    else:
        print("Please provide --images or use --random flag")
        print("Examples:")
        print("  python visualize.py --images dog1.jpg dog2.jpg dog3.jpg")
        print("  python visualize.py --random --num-samples 8")


if __name__ == "__main__":
    main()
