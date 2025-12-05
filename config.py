"""
Configuration file for dog breed classification training
Modify these settings to customize your training
"""

# Data Configuration
DATA_DIR = "images/Images"              # Path to images directory
TRAIN_SPLIT = 0.8                       # Fraction of data for training (0.8 = 80%)

# Training Configuration
BATCH_SIZE = 32                          # Batch size (reduce if out of memory)
NUM_EPOCHS = 20                          # Number of training epochs
LEARNING_RATE = 0.001                    # Initial learning rate
NUM_WORKERS = 4                          # Data loader workers (set to 0 if issues)

# Model Configuration
MODEL_NAME = "resnet50"                  # Base model architecture
PRETRAINED = True                        # Use ImageNet pretrained weights

# Optimization Configuration
OPTIMIZER = "adam"                       # Optimizer: adam, sgd, adamw
LR_SCHEDULER = "plateau"                 # LR scheduler: plateau, step, cosine
LR_FACTOR = 0.5                         # LR reduction factor
LR_PATIENCE = 3                         # Epochs with no improvement before LR reduction

# Augmentation Configuration
RANDOM_CROP = True                       # Random resized crop
RANDOM_FLIP = True                       # Random horizontal flip
RANDOM_ROTATION = 15                     # Random rotation degrees (0 to disable)
COLOR_JITTER = True                      # Apply color jitter
BRIGHTNESS = 0.2                         # Brightness jitter factor
CONTRAST = 0.2                           # Contrast jitter factor
SATURATION = 0.2                         # Saturation jitter factor

# Input Configuration
IMAGE_SIZE = 224                         # Input image size (224 for ResNet)
CROP_SIZE = 224                          # Crop size for center/random crop

# Normalization (ImageNet statistics)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Output Configuration
SAVE_DIR = "models"                      # Directory to save models
SAVE_BEST = True                         # Save model with best validation accuracy
SAVE_LAST = True                         # Save model after last epoch
PLOT_CURVES = True                       # Plot training curves

# Device Configuration
DEVICE = "cuda"                          # Device: cuda or cpu (auto-detected if cuda unavailable)

# Random Seed
RANDOM_SEED = 42                         # Random seed for reproducibility
