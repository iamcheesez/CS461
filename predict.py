"""
Inference script for dog breed classification
Use a trained ResNet50 model to predict dog breeds from images
"""
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import argparse
from pathlib import Path


class DogBreedPredictor:
    """Predict dog breeds from images using trained ResNet50 model"""
    
    def __init__(self, model_path: str, class_mapping_path: str, device: str = 'cuda'):
        """
        Args:
            model_path: Path to trained model checkpoint
            class_mapping_path: Path to class mapping JSON file
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load class mapping
        with open(class_mapping_path, 'r') as f:
            self.class_to_idx = json.load(f)
        
        self.idx_to_class = {idx: class_name for class_name, idx in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)
        print(f"Loaded {self.num_classes} breed classes")
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, model_path: str):
        """Load trained model from checkpoint"""
        # Create model architecture
        model = models.resnet50(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.num_classes)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        print(f"Model loaded from {model_path}")
        if 'val_acc' in checkpoint:
            print(f"Model validation accuracy: {checkpoint['val_acc']:.2f}%")
        
        return model
    
    def get_breed_name(self, folder_name: str) -> str:
        """Convert folder name to readable breed name"""
        breed_name = folder_name.split('-', 1)[1] if '-' in folder_name else folder_name
        return breed_name.replace('_', ' ')
    
    def predict(self, image_path: str, top_k: int = 5):
        """
        Predict dog breed from image
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
            
        Returns:
            List of tuples (breed_name, probability)
        """
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image: {e}")
            return []
        
        # Transform and add batch dimension
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        
        # Convert to breed names
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            folder_name = self.idx_to_class[idx]
            breed_name = self.get_breed_name(folder_name)
            predictions.append((breed_name, float(prob)))
        
        return predictions
    
    def predict_batch(self, image_paths: list, top_k: int = 5):
        """Predict breeds for multiple images"""
        results = {}
        for img_path in image_paths:
            print(f"\nPredicting: {img_path}")
            predictions = self.predict(img_path, top_k)
            results[img_path] = predictions
            
            if predictions:
                print(f"Top {top_k} predictions:")
                for i, (breed, prob) in enumerate(predictions, 1):
                    print(f"  {i}. {breed}: {prob * 100:.2f}%")
        
        return results


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(description='Dog Breed Classification Inference')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to image file or directory')
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--classes', type=str, default='models/class_mapping.json',
                       help='Path to class mapping JSON')
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of top predictions to show')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'], help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = DogBreedPredictor(
        model_path=args.model,
        class_mapping_path=args.classes,
        device=args.device
    )
    
    # Check if input is file or directory
    image_path = Path(args.image)
    
    if image_path.is_file():
        # Single image prediction
        print(f"\n{'=' * 60}")
        print(f"Predicting breed for: {image_path}")
        print('=' * 60)
        
        predictions = predictor.predict(str(image_path), top_k=args.top_k)
        
        if predictions:
            print(f"\nTop {args.top_k} predictions:")
            for i, (breed, prob) in enumerate(predictions, 1):
                print(f"  {i}. {breed:30s} {prob * 100:6.2f}%")
        else:
            print("Failed to make predictions")
    
    elif image_path.is_dir():
        # Batch prediction for all images in directory
        image_files = list(image_path.glob("*.jpg")) + list(image_path.glob("*.jpeg")) + \
                     list(image_path.glob("*.png")) + list(image_path.glob("*.JPEG"))
        
        if not image_files:
            print(f"No images found in {image_path}")
            return
        
        print(f"\n{'=' * 60}")
        print(f"Predicting breeds for {len(image_files)} images in {image_path}")
        print('=' * 60)
        
        predictor.predict_batch([str(f) for f in image_files], top_k=args.top_k)
    
    else:
        print(f"Error: {image_path} is not a valid file or directory")


if __name__ == "__main__":
    main()
