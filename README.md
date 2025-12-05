# Dog Breed Classification with ResNet50

A deep learning project for classifying 120 dog breeds using ResNet50 transfer learning on ImageNet dog breed dataset.

## Project Structure

```
CS461/
├── annotations/           # XML annotations (not used for classification)
├── images/               # Dog breed images organized by breed
│   └── Images/
│       ├── n02085620-Chihuahua/
│       ├── n02085782-Japanese_spaniel/
│       └── ... (120 breeds total)
├── models/               # Saved models and training artifacts (created during training)
├── dataset.py            # Dataset loader and preprocessing
├── train.py              # Model training script
├── predict.py            # Inference script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Dataset

- **120 dog breeds** from ImageNet
- Images organized in breed-specific folders
- Automatic 80/20 train/validation split
- Data augmentation applied during training

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

### Key Dependencies:
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- Pillow >= 9.0.0
- numpy, matplotlib, scikit-learn, tqdm

## Training the Model

### Basic Training

```bash
python train.py
```

This will:
1. Load and split the dataset (80% train, 20% validation)
2. Initialize ResNet50 with ImageNet pre-trained weights
3. Train for 20 epochs with data augmentation
4. Save the best model to `models/best_model.pth`
5. Save training history and curves

### Training Configuration

You can modify these parameters in `train.py`:

```python
DATA_DIR = "images/Images"      # Path to images folder
BATCH_SIZE = 32                  # Batch size (adjust based on GPU memory)
NUM_EPOCHS = 20                  # Number of training epochs
LEARNING_RATE = 0.001            # Initial learning rate
NUM_WORKERS = 4                  # Data loader workers
```

### Training Features

- **Transfer Learning**: Uses pre-trained ResNet50 from ImageNet
- **Data Augmentation**: Random crops, flips, rotation, color jitter
- **Learning Rate Scheduling**: Reduces LR on plateau
- **Model Checkpointing**: Saves best and latest models
- **Training Visualization**: Plots loss and accuracy curves

### Expected Training Time

- **GPU (NVIDIA RTX 3080)**: ~2-3 hours for 20 epochs
- **GPU (NVIDIA T4)**: ~4-5 hours for 20 epochs
- **CPU**: Not recommended (very slow)

## Model Architecture

- **Base**: ResNet50 pre-trained on ImageNet
- **Modified**: Final fully connected layer replaced for 120-class classification
- **Input Size**: 224x224 RGB images
- **Output**: Probability distribution over 120 dog breeds

## Making Predictions

### Predict Single Image

```bash
python predict.py --image path/to/dog_image.jpg
```

### Predict Multiple Images

```bash
python predict.py --image path/to/image_folder/
```

### Prediction Options

```bash
python predict.py \
    --image path/to/image.jpg \
    --model models/best_model.pth \
    --classes models/class_mapping.json \
    --top-k 5 \
    --device cuda
```

**Arguments:**
- `--image`: Path to image file or directory (required)
- `--model`: Path to trained model checkpoint (default: `models/best_model.pth`)
- `--classes`: Path to class mapping JSON (default: `models/class_mapping.json`)
- `--top-k`: Number of top predictions to show (default: 5)
- `--device`: Device to use (`cuda` or `cpu`, default: `cuda`)

### Example Output

```
Using device: cuda
Loaded 120 breed classes
Model loaded from models/best_model.pth
Model validation accuracy: 85.42%

============================================================
Predicting breed for: test_image.jpg
============================================================

Top 5 predictions:
  1. golden retriever              94.32%
  2. Labrador retriever             3.21%
  3. flat-coated retriever          1.15%
  4. curly-coated retriever         0.82%
  5. Chesapeake Bay retriever       0.31%
```

## Training Results

After training, you'll find these files in the `models/` directory:

- `best_model.pth`: Model checkpoint with best validation accuracy
- `latest_model.pth`: Latest model checkpoint
- `class_mapping.json`: Mapping between class indices and breed names
- `training_history.json`: Loss and accuracy for each epoch
- `training_curves.png`: Visualization of training progress

## Data Augmentation

### Training Augmentations:
- Random resized crop (224x224)
- Random horizontal flip
- Random rotation (±15 degrees)
- Color jitter (brightness, contrast, saturation)
- ImageNet normalization

### Validation/Inference:
- Resize to 256x256
- Center crop to 224x224
- ImageNet normalization

## Performance Tips

1. **GPU Memory**: If you get out-of-memory errors, reduce `BATCH_SIZE` in `train.py`
2. **Training Speed**: Increase `NUM_WORKERS` if you have a fast CPU and slow GPU
3. **Accuracy**: Train for more epochs or fine-tune the learning rate
4. **Overfitting**: Add dropout or increase data augmentation

## Troubleshooting

### Import Errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### CUDA Out of Memory
Reduce batch size in `train.py`:
```python
BATCH_SIZE = 16  # or 8
```

### Slow Training
- Use GPU if available
- Increase `NUM_WORKERS` for data loading
- Use smaller image size (modify transforms)

### Low Accuracy
- Train for more epochs
- Adjust learning rate
- Try different data augmentation strategies
- Fine-tune more layers of ResNet50

## Dog Breeds Included

The model can classify 120 dog breeds including:
- Chihuahua, Japanese spaniel, Maltese dog, Pekinese, Shih-Tzu
- Beagle, Bloodhound, Golden retriever, Labrador retriever
- German shepherd, Rottweiler, Doberman, Boxer
- Husky, Malamute, Saint Bernard, Great Dane
- And 100+ more breeds...

## License

This project uses the ImageNet dataset which has its own terms of use. Please ensure compliance with ImageNet's terms when using this code.

## Future Improvements

- [ ] Add confusion matrix visualization
- [ ] Implement K-fold cross-validation
- [ ] Try other architectures (EfficientNet, Vision Transformer)
- [ ] Add Grad-CAM visualization for interpretability
- [ ] Web interface for easy predictions
- [ ] Mobile deployment (TensorFlow Lite, ONNX)
