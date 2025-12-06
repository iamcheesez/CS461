"""
What DAWG!? - Dog Breed Classification Web Application
Flask app that uses ResNet50 model to classify dog breeds from uploaded images.
Supports bilingual (Thai/English) breed information.
"""

import os
import json
import uuid
import shutil
import secrets
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# =============================================================================
# Configuration
# =============================================================================

app = Flask(__name__, static_folder='static')

# Security
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))

# Paths (configurable via environment variables)
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
FEEDBACK_FOLDER = os.environ.get('FEEDBACK_FOLDER', 'feedback_data')
FEEDBACK_LOG = os.path.join(FEEDBACK_FOLDER, 'feedback_log.json')
BREED_INFO_EN_FILE = os.environ.get('BREED_INFO_EN', 'breed_info_en.json')
BREED_INFO_TH_FILE = os.environ.get('BREED_INFO_TH', 'breed_info_th.json')
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/best_model.pth')
CLASS_MAPPING_PATH = os.environ.get('CLASS_MAPPING_PATH', 'models/class_mapping.json')

# Settings
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create required folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FEEDBACK_FOLDER, exist_ok=True)


# =============================================================================
# Breed Information (Bilingual Support)
# =============================================================================

breed_info_cache_en = {}
breed_info_cache_th = {}
current_language = 'th'


def load_breed_info_from_file(filepath):
    """Load breed information from JSON file into cache."""
    cache = {}
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for breed in data.get('data', []):
                attrs = breed.get('attributes', {})
                name = attrs.get('name', '').lower()
                cache[name] = {
                    'id': breed.get('id'),
                    'name': attrs.get('name'),
                    'description': attrs.get('description'),
                    'life': attrs.get('life', {}),
                    'male_weight': attrs.get('male_weight', {}),
                    'female_weight': attrs.get('female_weight', {}),
                    'hypoallergenic': attrs.get('hypoallergenic', False)
                }
            print(f"Loaded {len(cache)} breeds from {filepath}")
        else:
            print(f"Warning: File not found: {filepath}")
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
    return cache


def load_all_breed_info():
    """Load breed info for all supported languages."""
    global breed_info_cache_en, breed_info_cache_th
    print("Loading breed info...")
    breed_info_cache_en = load_breed_info_from_file(BREED_INFO_EN_FILE)
    breed_info_cache_th = load_breed_info_from_file(BREED_INFO_TH_FILE)
    print(f"Breed info loaded: EN={len(breed_info_cache_en)}, TH={len(breed_info_cache_th)}")


def get_breed_info(breed_name, lang=None):
    """Get breed information from cache based on language."""
    if lang is None:
        lang = current_language
    
    cache = breed_info_cache_th if lang == 'th' else breed_info_cache_en
    
    # Fallback to other language if empty
    if not cache:
        cache = breed_info_cache_en if lang == 'th' else breed_info_cache_th
    
    if not cache:
        return None
    
    # Try exact match
    name_lower = breed_name.lower()
    if name_lower in cache:
        return cache[name_lower]
    
    # Try partial match
    for key, info in cache.items():
        if name_lower in key or key in name_lower:
            return info
        # Word intersection match
        breed_words = set(name_lower.replace('-', ' ').replace('_', ' ').split())
        key_words = set(key.replace('-', ' ').replace('_', ' ').split())
        if breed_words & key_words:
            return info
    
    return None


# =============================================================================
# Dog Breed Predictor (Singleton)
# =============================================================================

class DogBreedPredictor:
    """Singleton class for dog breed prediction using ResNet50."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.model_path = MODEL_PATH
        self.class_mapping_path = CLASS_MAPPING_PATH
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load class mapping
        with open(self.class_mapping_path, 'r') as f:
            self.class_to_idx = json.load(f)
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        self.num_classes = len(self.class_to_idx)
        print(f"Loaded {self.num_classes} breed classes")
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        self._initialized = True
    
    def _load_model(self):
        """Load trained ResNet50 model."""
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        print(f"Model loaded from {self.model_path}")
        return model
    
    def get_breed_name(self, folder_name):
        """Convert folder name to readable breed name."""
        breed_name = folder_name.split('-', 1)[1] if '-' in folder_name else folder_name
        return breed_name.replace('_', ' ').title()
    
    def predict(self, image_path, top_k=5):
        """Predict dog breed from image."""
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image: {e}")
            return []
        
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        top_probs, top_indices = torch.topk(probabilities, top_k)
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]
        
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            folder_name = self.idx_to_class[idx]
            predictions.append({
                'breed': self.get_breed_name(folder_name),
                'probability': float(prob),
                'percentage': f"{float(prob) * 100:.2f}%"
            })
        
        return predictions


# Singleton instance
predictor = None


def get_predictor():
    """Get or create predictor instance."""
    global predictor
    if predictor is None:
        predictor = DogBreedPredictor()
    return predictor


# =============================================================================
# Helper Functions
# =============================================================================

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def log_feedback(entry):
    """Append feedback entry to log file."""
    logs = []
    if os.path.exists(FEEDBACK_LOG):
        try:
            with open(FEEDBACK_LOG, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        except:
            logs = []
    
    logs.append(entry)
    
    with open(FEEDBACK_LOG, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)


# =============================================================================
# Routes - Main Pages
# =============================================================================

@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'model_loaded': predictor is not None})


# =============================================================================
# Routes - Prediction
# =============================================================================

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and return predictions."""
    if 'file' not in request.files:
        return jsonify({'error': 'ไม่พบไฟล์รูปภาพ'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'กรุณาเลือกไฟล์'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'ไฟล์ไม่รองรับ กรุณาใช้ไฟล์ .jpg, .jpeg, .png, .gif หรือ .webp'}), 400
    
    try:
        # Save file
        filename = secure_filename(file.filename)
        image_id = uuid.uuid4().hex
        unique_filename = f"{image_id}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        
        # Get predictions
        pred = get_predictor()
        predictions = pred.predict(filepath, top_k=5)
        
        if not predictions:
            os.remove(filepath)
            return jsonify({'error': 'ไม่สามารถวิเคราะห์รูปภาพได้'}), 500
        
        # Add class keys and breed info
        for p in predictions:
            for folder_name, idx in pred.class_to_idx.items():
                if pred.get_breed_name(folder_name) == p['breed']:
                    p['class_key'] = folder_name
                    break
            
            breed_info = get_breed_info(p['breed'])
            if breed_info:
                p['info'] = breed_info
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'top_breed': predictions[0]['breed'],
            'top_confidence': predictions[0]['percentage'],
            'top_breed_info': get_breed_info(predictions[0]['breed']),
            'image_id': image_id,
            'image_filename': unique_filename
        })
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': f'เกิดข้อผิดพลาด: {str(e)}'}), 500


# =============================================================================
# Routes - Feedback System
# =============================================================================

@app.route('/feedback', methods=['POST'])
def feedback():
    """Handle user feedback on predictions."""
    try:
        data = request.get_json()
        
        image_id = data.get('image_id')
        image_filename = data.get('image_filename')
        is_correct = data.get('is_correct')
        predicted_breed = data.get('predicted_breed')
        predicted_class_key = data.get('predicted_class_key')
        correct_breed = data.get('correct_breed')
        correct_class_key = data.get('correct_class_key')
        
        if not image_id or not image_filename:
            return jsonify({'error': 'Missing image information'}), 400
        
        src_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
        
        if not os.path.exists(src_path):
            return jsonify({'error': 'Image not found'}), 404
        
        # Determine target class
        if is_correct:
            target_class = predicted_class_key
            target_breed = predicted_breed
        else:
            target_class = correct_class_key
            target_breed = correct_breed
        
        # Create feedback entry
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'image_id': image_id,
            'original_filename': image_filename,
            'is_correct': is_correct,
            'predicted_breed': predicted_breed,
            'predicted_class_key': predicted_class_key,
            'correct_breed': target_breed,
            'correct_class_key': target_class,
            'saved_for_training': False
        }
        
        # Save image for training if we have target class
        if target_class:
            class_folder = os.path.join(FEEDBACK_FOLDER, 'images', target_class)
            os.makedirs(class_folder, exist_ok=True)
            
            dest_filename = f"{image_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            dest_path = os.path.join(class_folder, dest_filename)
            shutil.copy2(src_path, dest_path)
            
            feedback_entry['saved_for_training'] = True
            feedback_entry['training_path'] = dest_path
        
        os.remove(src_path)
        log_feedback(feedback_entry)
        
        return jsonify({
            'success': True,
            'message': 'ขอบคุณสำหรับ feedback! ข้อมูลจะถูกนำไปปรับปรุง AI',
            'saved_for_training': feedback_entry['saved_for_training']
        })
        
    except Exception as e:
        print(f"Error saving feedback: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/skip_feedback', methods=['POST'])
def skip_feedback():
    """Clean up image when user skips feedback."""
    try:
        data = request.get_json()
        image_filename = data.get('image_filename')
        
        if image_filename:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
            if os.path.exists(filepath):
                os.remove(filepath)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/feedback_stats')
def feedback_stats():
    """Get feedback statistics."""
    try:
        if not os.path.exists(FEEDBACK_LOG):
            return jsonify({
                'total_feedback': 0,
                'correct_predictions': 0,
                'incorrect_predictions': 0,
                'accuracy': "N/A",
                'saved_for_training': 0
            })
        
        with open(FEEDBACK_LOG, 'r', encoding='utf-8') as f:
            logs = json.load(f)
        
        total = len(logs)
        correct = sum(1 for log in logs if log.get('is_correct'))
        
        return jsonify({
            'total_feedback': total,
            'correct_predictions': correct,
            'incorrect_predictions': total - correct,
            'accuracy': f"{(correct/total*100):.1f}%" if total > 0 else "N/A",
            'saved_for_training': sum(1 for log in logs if log.get('saved_for_training'))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Routes - Breed Information & Language
# =============================================================================

@app.route('/breeds')
def get_breeds():
    """Get list of all available breeds."""
    pred = get_predictor()
    breeds = [
        {'class_key': name, 'breed_name': pred.get_breed_name(name)}
        for name in sorted(pred.class_to_idx.keys())
    ]
    return jsonify({'breeds': breeds})


@app.route('/breed-info/<breed_name>')
def get_breed_info_api(breed_name):
    """Get breed information by name."""
    info = get_breed_info(breed_name)
    if info:
        return jsonify({'success': True, 'breed_info': info})
    return jsonify({'success': False, 'error': 'Breed not found'}), 404


@app.route('/language', methods=['GET'])
def get_language():
    """Get current language setting."""
    return jsonify({'language': current_language, 'available': ['en', 'th']})


@app.route('/language', methods=['POST'])
def set_language():
    """Set language for breed information."""
    global current_language
    try:
        data = request.get_json()
        lang = data.get('language', 'th').lower()
        
        if lang not in ['en', 'th']:
            return jsonify({'error': 'Unsupported language'}), 400
        
        current_language = lang
        return jsonify({
            'success': True,
            'language': current_language,
            'message': f'Language: {"ไทย" if lang == "th" else "English"}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================================================================
# Health Check Endpoint (for deployment)
# =============================================================================

@app.route('/health')
def health_check():
    """Health check endpoint for load balancers and monitoring."""
    try:
        # Check if model is loaded
        predictor = get_predictor()
        model_loaded = predictor is not None and predictor.model is not None
        
        return jsonify({
            'status': 'healthy',
            'model_loaded': model_loaded,
            'breeds_en': len(breed_info_cache_en),
            'breeds_th': len(breed_info_cache_th),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == '__main__':
    load_all_breed_info()
    print("Loading model...")
    get_predictor()
    print("Model loaded! Starting server...")
    
    # Get port from environment (for cloud deployment)
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    app.run(debug=debug, host='0.0.0.0', port=port)
