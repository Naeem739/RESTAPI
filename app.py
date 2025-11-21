from flask import Flask, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import base64
import json
import uuid

# Handle different TensorFlow/Keras versions
try:
    # Try tf.keras first (TensorFlow 2.x)
    from tensorflow import keras
    print(f"Using TensorFlow {tf.__version__} with tf.keras")
except ImportError:
    try:
        # Fallback to standalone keras
        import keras
        print(f"Using standalone Keras {keras.__version__}")
    except ImportError:
        print("❌ Neither tf.keras nor standalone keras is available")
        keras = None

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///paddy_disease.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
CORS(app)

# File uploads directory for history images
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Serve uploaded files
@app.route('/uploads/<path:filename>')
def serve_uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# Load the paddy disease detection model (neuralnetwork.keras)
def _load_model_with_fallbacks():
    """Load Keras model trying several likely paths."""
    possible_paths = []
    # 1) Same directory as this file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths.append(os.path.join(base_dir, 'neuralnetwork.keras'))
    # 2) Working directory
    possible_paths.extend([
        'neuralnetwork.keras',
        './neuralnetwork.keras'
    ])
    # 3) Legacy name fallback
    possible_paths.append(os.path.join(base_dir, 'model.h5'))

    last_err = None
    for p in possible_paths:
        try:
            if os.path.exists(p):
                print(f"Trying to load model from: {p}")
            model_local = keras.models.load_model(p)
            print("Paddy disease detection model loaded successfully!")
            return model_local
        except Exception as e:  # keep trying others
            last_err = e
            continue
    print(f"Error loading model: {last_err}")
    return None

model = _load_model_with_fallbacks()

# Define disease classes (10-class model as per notebook)
DISEASE_CLASSES = [
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "blast",
    "brown_spot",
    "dead_heart",
    "downy_mildew",
    "hispa",
    "normal",
    "tungro",
]

# Disease information database
DISEASE_INFO = {
    'brown_spot': {
        'description': 'A fungal disease caused by Cochliobolus miyabeanus that produces oval or circular brown lesions on leaves, sheaths, and grains. The spots have distinct brown margins and can cause significant yield loss.',
        'treatment': 'Use disease-free seeds, maintain balanced nutrition, avoid excessive nitrogen, and apply fungicides like mancozeb or carbendazim. Remove infected plant debris and maintain proper field hygiene.'
    },
    'bacterial_leaf_blight': {
        'description': 'A serious bacterial disease caused by Xanthomonas oryzae pv. oryzae. It causes wilting of seedlings and yellowing and drying of leaves. The disease spreads rapidly in warm, humid conditions.',
        'treatment': 'Use resistant varieties, avoid excessive nitrogen fertilization, maintain proper field drainage, apply copper-based bactericides, and remove infected plants early to prevent spread.'
    },
    'blast': {
        'description': 'A common rice disease caused by Magnaporthe oryzae, leading to spindle-shaped lesions on leaves and potentially severe yield loss.',
        'treatment': 'Use resistant varieties, balanced fertilization, and apply recommended fungicides when necessary.'
    }
}

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# Notice Model
class Notice(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

# Prediction History Model
class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_url = db.Column(db.String(500), default='')
    disease_name = db.Column(db.String(120), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    description = db.Column(db.Text, default='')
    treatment = db.Column(db.Text, default='')
    search_time = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    # Store JSON as text for SQLite compatibility
    full_result = db.Column(db.Text, default='{}')

# Ensure all tables exist (including History)
with app.app_context():
    db.create_all()

# Create database tables
with app.app_context():
    db.create_all()

# Helper function to generate JWT token
def generate_token(user_id):
    payload = {
        'user_id': user_id,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(days=1)
    }
    return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')

# Helper function to verify JWT token
def verify_token(token):
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        return payload['user_id']
    except:
        return None

# Authentication decorator
def token_required(f):
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        
        token = token.split(' ')[1] if token.startswith('Bearer ') else token
        user_id = verify_token(token)
        if not user_id:
            return jsonify({'message': 'Token is invalid'}), 401
        
        return f(user_id, *args, **kwargs)
    return decorated

# Helper function to preprocess image for model
def preprocess_image(image_file):
    try:
        # Open and resize image to notebook's target size (224x224)
        image = Image.open(image_file).convert('RGB')

        target_size = (224, 224)
        try:
            # If model has a specific input size, prefer that
            if hasattr(model, 'input_shape') and model.input_shape is not None:
                _, h, w, c = model.input_shape
                if h is not None and w is not None and c in (1, 3):
                    target_size = (w, h) if isinstance(w, int) and isinstance(h, int) else target_size
        except Exception:
            pass

        resized_image = image.resize(target_size, Image.BILINEAR)

        # Convert to numpy array (no manual normalization; assume model/rescaling handles it)
        image_array = keras.utils.img_to_array(resized_image)
        image_batch = np.expand_dims(image_array, axis=0)

        return image_batch

    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# Routes

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Paddy Disease Detector API is running'})

# User Registration
@app.route('/api/auth/signup', methods=['POST'])
def signup():
    data = request.get_json()
    
    if not data or not data.get('username') or not data.get('email') or not data.get('password'):
        return jsonify({'message': 'Missing required fields'}), 400
    
    # Check if user already exists
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'message': 'Username already exists'}), 400
    
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'message': 'Email already exists'}), 400
    
    # Create new user
    hashed_password = generate_password_hash(data['password'])
    new_user = User(
        username=data['username'],
        email=data['email'],
        password_hash=hashed_password
    )
    
    try:
        db.session.add(new_user)
        db.session.commit()
        
        token = generate_token(new_user.id)
        return jsonify({
            'message': 'User created successfully',
            'token': token,
            'user': {
                'id': new_user.id,
                'username': new_user.username,
                'email': new_user.email,
                'created_at': new_user.created_at.isoformat()
            }
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': 'Error creating user'}), 500

# User Login
@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json()
    
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'message': 'Missing username or password'}), 400
    
    user = User.query.filter_by(username=data['username']).first()
    
    if user and check_password_hash(user.password_hash, data['password']):
        token = generate_token(user.id)
        return jsonify({
            'message': 'Login successful',
            'token': token,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'created_at': user.created_at.isoformat()
            }
        }), 200
    else:
        return jsonify({'message': 'Invalid username or password'}), 401

# Get User Profile
@app.route('/api/user/profile', methods=['GET'], endpoint='get_profile')
@token_required
def get_profile(user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({'message': 'User not found'}), 404
    
    return jsonify({
        'id': user.id,
        'username': user.username,
        'email': user.email,
        'created_at': user.created_at.isoformat()
    }), 200

# Get All Users (Admin endpoint)
@app.route('/api/users', methods=['GET'], endpoint='get_all_users')
def get_all_users():
    try:
        users = User.query.all()
        users_list = []
        for user in users:
            users_list.append({
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'created_at': user.created_at.isoformat()
            })
        
        return jsonify({'users': users_list}), 200
    except Exception as e:
        return jsonify({'message': 'Error fetching users'}), 500

# Notice CRUD Operations

# Create Notice
@app.route('/api/notices', methods=['POST'], endpoint='create_notice')
@token_required
def create_notice(user_id):
    data = request.get_json()
    
    if not data or not data.get('title') or not data.get('content'):
        return jsonify({'message': 'Missing title or content'}), 400
    
    new_notice = Notice(
        title=data['title'],
        content=data['content'],
        user_id=user_id
    )
    
    try:
        db.session.add(new_notice)
        db.session.commit()
        
        return jsonify({
            'message': 'Notice created successfully',
            'notice': {
                'id': new_notice.id,
                'title': new_notice.title,
                'content': new_notice.content,
                'user_id': new_notice.user_id,
                'created_at': new_notice.created_at.isoformat()
            }
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': 'Error creating notice'}), 500

# Get All Notices
@app.route('/api/notices', methods=['GET'], endpoint='get_notices')
@token_required
def get_notices(user_id):
    notices = Notice.query.filter_by(user_id=user_id).order_by(Notice.created_at.desc()).all()
    
    notices_list = []
    for notice in notices:
        notices_list.append({
            'id': notice.id,
            'title': notice.title,
            'content': notice.content,
            'user_id': notice.user_id,
            'created_at': notice.created_at.isoformat(),
            'updated_at': notice.updated_at.isoformat()
        })
    
    return jsonify({'notices': notices_list}), 200

# Get Single Notice
@app.route('/api/notices/<int:notice_id>', methods=['GET'], endpoint='get_notice')
@token_required
def get_notice(user_id, notice_id):
    notice = Notice.query.filter_by(id=notice_id, user_id=user_id).first()
    
    if not notice:
        return jsonify({'message': 'Notice not found'}), 404
    
    return jsonify({
        'id': notice.id,
        'title': notice.title,
        'content': notice.content,
        'user_id': notice.user_id,
        'created_at': notice.created_at.isoformat(),
        'updated_at': notice.updated_at.isoformat()
    }), 200

# Update Notice
@app.route('/api/notices/<int:notice_id>', methods=['PUT'], endpoint='update_notice')
@token_required
def update_notice(user_id, notice_id):
    notice = Notice.query.filter_by(id=notice_id, user_id=user_id).first()
    
    if not notice:
        return jsonify({'message': 'Notice not found'}), 404
    
    data = request.get_json()
    
    if data.get('title'):
        notice.title = data['title']
    if data.get('content'):
        notice.content = data['content']
    
    notice.updated_at = datetime.datetime.utcnow()
    
    try:
        db.session.commit()
        
        return jsonify({
            'message': 'Notice updated successfully',
            'notice': {
                'id': notice.id,
                'title': notice.title,
                'content': notice.content,
                'user_id': notice.user_id,
                'created_at': notice.created_at.isoformat(),
                'updated_at': notice.updated_at.isoformat()
            }
        }), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': 'Error updating notice'}), 500

# Delete Notice
@app.route('/api/notices/<int:notice_id>', methods=['DELETE'], endpoint='delete_notice')
@token_required
def delete_notice(user_id, notice_id):
    notice = Notice.query.filter_by(id=notice_id, user_id=user_id).first()
    
    if not notice:
        return jsonify({'message': 'Notice not found'}), 404
    
    try:
        db.session.delete(notice)
        db.session.commit()
        
        return jsonify({'message': 'Notice deleted successfully'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'message': 'Error deleting notice'}), 500

# Prediction History Endpoints

# Get history for current user
@app.route('/api/history', methods=['GET'], endpoint='get_history')
@token_required
def get_history(user_id):
    try:
        items = History.query.filter_by(user_id=user_id).order_by(History.search_time.desc()).all()
        history_list = []
        for item in items:
            try:
                full_result = json.loads(item.full_result) if item.full_result else {}
            except Exception:
                full_result = {}
            history_list.append({
                'id': item.id,
                'image_url': item.image_url or '',
                'disease_name': item.disease_name,
                'confidence': float(item.confidence),
                'description': item.description or '',
                'treatment': item.treatment or '',
                'search_time': item.search_time.isoformat(),
                'full_result': full_result
            })
        return jsonify({'history': history_list}), 200
    except Exception:
        return jsonify({'message': 'Error fetching history'}), 500

# Add new history item
@app.route('/api/history', methods=['POST'], endpoint='create_history')
@token_required
def create_history(user_id):
    data = request.get_json() or {}
    required_fields = ['disease_name', 'confidence']
    if any(f not in data for f in required_fields):
        return jsonify({'message': 'Missing required fields'}), 400

    try:
        # Handle optional base64 image upload
        image_url_value = data.get('image_url', '') or ''
        image_base64 = data.get('image_base64')
        if image_base64:
            try:
                # Support possible data URL prefix
                if ',' in image_base64:
                    image_base64 = image_base64.split(',', 1)[1]
                image_bytes = base64.b64decode(image_base64)
                filename = f"{uuid.uuid4().hex}.jpg"
                file_path = os.path.join(UPLOAD_FOLDER, filename)
                with open(file_path, 'wb') as f:
                    f.write(image_bytes)
                # Build absolute URL to the file
                base = request.host_url.rstrip('/')
                image_url_value = f"{base}/uploads/{filename}"
            except Exception:
                # If image fails to save, continue without blocking
                image_url_value = ''

        new_item = History(
            user_id=user_id,
            image_url=image_url_value,
            disease_name=data['disease_name'],
            confidence=float(data.get('confidence', 0.0)),
            description=data.get('description', '') or '',
            treatment=data.get('treatment', '') or '',
            search_time=datetime.datetime.utcnow(),
            full_result=json.dumps(data.get('full_result', {}))
        )
        db.session.add(new_item)
        db.session.commit()

        return jsonify({
            'message': 'History item created successfully',
            'history_item': {
                'id': new_item.id,
                'image_url': new_item.image_url,
                'disease_name': new_item.disease_name,
                'confidence': float(new_item.confidence),
                'description': new_item.description,
                'treatment': new_item.treatment,
                'search_time': new_item.search_time.isoformat(),
                'full_result': json.loads(new_item.full_result) if new_item.full_result else {}
            }
        }), 201
    except Exception:
        db.session.rollback()
        return jsonify({'message': 'Error creating history item'}), 500

# Delete a single history item
@app.route('/api/history/<int:item_id>', methods=['DELETE'], endpoint='delete_history_item')
@token_required
def delete_history_item(user_id, item_id):
    try:
        item = History.query.filter_by(id=item_id, user_id=user_id).first()
        if not item:
            return jsonify({'message': 'History item not found'}), 404
        db.session.delete(item)
        db.session.commit()
        return jsonify({'message': 'History item deleted successfully'}), 200
    except Exception:
        db.session.rollback()
        return jsonify({'message': 'Error deleting history item'}), 500

# Clear all history for the current user
@app.route('/api/history/clear', methods=['DELETE'], endpoint='clear_history')
@token_required
def clear_history(user_id):
    try:
        History.query.filter_by(user_id=user_id).delete()
        db.session.commit()
        return jsonify({'message': 'All history cleared successfully'}), 200
    except Exception:
        db.session.rollback()
        return jsonify({'message': 'Error clearing history'}), 500

# Paddy Disease Prediction Endpoint
@app.route('/api/predict/paddy-disease', methods=['POST'], endpoint='predict_paddy_disease')
@token_required
def predict_paddy_disease(user_id):
    if model is None:
        return jsonify({'message': 'Model not available'}), 500
    
    if 'image' not in request.files:
        return jsonify({'message': 'No image file provided'}), 400
    
    try:
        image_file = request.files['image']
        
        if image_file.filename == '':
            return jsonify({'message': 'No image file selected'}), 400
        
        # Print model information for debugging
        print(f"🔍 Model input shape: {model.input_shape}")
        print(f"🔍 Model output shape: {model.output_shape}")
        
        # Preprocess the image
        processed_image = preprocess_image(image_file)
        if processed_image is None:
            return jsonify({'message': 'Error processing image'}), 400
        
        print(f"🔍 Processed image shape: {processed_image.shape}")
        
        # Make prediction
        try:
            raw_pred = model.predict(processed_image, verbose=0)[0]
            print(f"✅ Prediction successful, output shape: {(1, raw_pred.shape[0])}")

            # Ensure probabilities using softmax if needed (like notebook)
            if raw_pred.ndim == 1 and (raw_pred.min() < 0 or raw_pred.max() > 1.0 or not np.isclose(raw_pred.sum(), 1.0, atol=1e-3)):
                pred = tf.nn.softmax(raw_pred).numpy()
            else:
                pred = raw_pred

            # Validate number of classes
            actual_classes = pred.shape[0]
            expected_classes = len(DISEASE_CLASSES)
            if actual_classes != expected_classes:
                print(f"⚠️  Model output mismatch: Expected {expected_classes} classes, got {actual_classes}")
                return jsonify({
                    'message': f'Model output mismatch. Expected {expected_classes} classes but model outputs {actual_classes} classes.',
                    'error_details': f'Model output length: {actual_classes}, Expected: {expected_classes} classes',
                    'suggestion': 'Please update DISEASE_CLASSES to match your model output or retrain the model with the expected number of classes.'
                }), 400

        except Exception as prediction_error:
            print(f"❌ Prediction error: {prediction_error}")

            error_msg = str(prediction_error)
            if "expected axis -1 of input shape to have value" in error_msg:
                return jsonify({
                    'message': 'Model input size mismatch. The model expects a different image size than what was provided.',
                    'error_details': error_msg,
                    'suggestion': 'Please ensure the image is properly resized or contact support for model compatibility issues.'
                }), 400
            elif "index" in error_msg and "out of bounds" in error_msg:
                return jsonify({
                    'message': 'Model output size mismatch. The model outputs fewer classes than expected.',
                    'error_details': error_msg,
                    'suggestion': 'Please check the number of classes in your model or update the DISEASE_CLASSES list.'
                }), 400
            else:
                return jsonify({
                    'message': f'Model prediction failed: {error_msg}',
                    'suggestion': 'Please try with a different image or contact support.'
                }), 500

        # Top-3 predictions
        top3_idx = np.argsort(pred)[-3:][::-1]
        top3 = [
            {
                'class': DISEASE_CLASSES[i],
                'probability': float(pred[i]),
                'percent': float(pred[i] * 100.0)
            }
            for i in top3_idx
        ]

        # Primary prediction
        pred_idx = int(np.argmax(pred))
        confidence = float(np.max(pred))
        disease_name = DISEASE_CLASSES[pred_idx]
        disease_info = DISEASE_INFO.get(disease_name, {})

        # Format response similar to notebook output structure
        result = {
            'disease_name': disease_name,
            'confidence': confidence,
            'top3': top3,
            'description': disease_info.get('description', 'No description available'),
            'treatment': disease_info.get('treatment', 'No treatment information available'),
            'all_predictions': {
                disease: float(pred[i])
                for i, disease in enumerate(DISEASE_CLASSES)
            }
        }

        return jsonify({
            'message': 'Prediction successful',
            'result': result
        }), 200
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'message': f'Prediction error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False  # disable debug for production
    )

