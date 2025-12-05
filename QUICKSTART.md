# Dog Breed Classification - Quick Start Guide

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify dataset:**
   ```bash
   python test_dataset.py
   ```
   This should show:
   - ✓ Found 120 breed folders
   - ✓ Successfully loaded batch
   - ✓ Dataset test PASSED!

## Training

### Start Training
```bash
python train.py
```

**What happens during training:**
- Loads 120 dog breeds from `images/Images/`
- Splits data: 80% training, 20% validation
- Trains ResNet50 for 20 epochs
- Saves best model to `models/best_model.pth`
- Creates training curves in `models/training_curves.png`

**Training time:**
- GPU (RTX 3080): ~2-3 hours
- GPU (T4): ~4-5 hours  
- CPU: Not recommended

### Monitor Training
Watch the progress in terminal:
```
Epoch 1/20
Training: 100%|████████| Loss: 2.5432, Acc: 45.23%
Validation: 100%|████████| Loss: 2.1234, Acc: 52.45%
✓ Saved best model (Val Acc: 52.45%)
```

## Making Predictions

### Predict Single Image
```bash
python predict.py --image path/to/dog.jpg
```

Output:
```
Top 5 predictions:
  1. golden retriever              94.32%
  2. Labrador retriever             3.21%
  3. flat-coated retriever          1.15%
```

### Predict Multiple Images
```bash
python predict.py --image path/to/folder/
```

### Visualize Predictions
```bash
# Random samples from dataset
python visualize.py --random --num-samples 8

# Specific images
python visualize.py --images dog1.jpg dog2.jpg dog3.jpg
```

## Common Issues

### "No module named 'torch'"
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### "CUDA out of memory"
Edit `train.py`, line 238:
```python
BATCH_SIZE = 16  # Reduce from 32
```

### "Images directory not found"
Ensure folder structure:
```
CS461/
├── images/
│   └── Images/
│       ├── n02085620-Chihuahua/
│       ├── n02085782-Japanese_spaniel/
│       └── ...
```

## File Overview

| File | Purpose |
|------|---------|
| `dataset.py` | Loads and preprocesses images |
| `train.py` | Trains ResNet50 model |
| `predict.py` | Makes predictions on new images |
| `visualize.py` | Visualizes predictions with plots |
| `test_dataset.py` | Verifies dataset is loaded correctly |
| `requirements.txt` | Python dependencies |

## Next Steps

1. **Test dataset**: `python test_dataset.py`
2. **Train model**: `python train.py`
3. **Make predictions**: `python predict.py --image your_dog.jpg`
4. **Visualize results**: `python visualize.py --random`

## Expected Results

After 20 epochs of training:
- **Training Accuracy**: 85-95%
- **Validation Accuracy**: 75-85%
- **Top-5 Accuracy**: 90-95%

The model should correctly identify most common breeds like:
- Golden Retriever
- German Shepherd
- Labrador Retriever
- Husky
- Beagle
- And 115+ more!

## Need Help?

Check `README.md` for detailed documentation and troubleshooting.
