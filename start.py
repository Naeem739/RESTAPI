#!/usr/bin/env python3
"""
Startup script for the Paddy Disease Detection Flask API.
This script checks dependencies and starts the server.
"""

import sys
import os
import subprocess
import importlib

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        'flask',
        'flask_sqlalchemy', 
        'flask_cors',
        'jwt',
        'numpy',
        'PIL',
        'tensorflow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                importlib.import_module('PIL')
            elif package == 'jwt':
                importlib.import_module('jwt')
            else:
                importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    return True

def check_model_file():
    """Check if the neuralnetwork.keras file exists (with fallbacks)."""
    candidate_paths = [
        'neuralnetwork.keras',
        './neuralnetwork.keras',
        'model.h5'
    ]
    for model_path in candidate_paths:
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # Size in MB
            print(f"✅ Model file found: {model_path} ({file_size:.1f} MB)")
            return True
    print("❌ Model file not found: neuralnetwork.keras or model.h5")
    print("Please place your trained model file in this directory")
    return False

def test_model_loading():
    """Test if the model can be loaded."""
    try:
        import tensorflow as tf
        print("🔄 Testing model loading...")
        for candidate in ['neuralnetwork.keras', './neuralnetwork.keras', 'model.h5']:
            try:
                model = tf.keras.models.load_model(candidate)
                print(f"✅ Model loaded successfully from {candidate}!")
                input_shape = model.input_shape
                output_shape = model.output_shape
                print(f"   Input shape: {input_shape}")
                print(f"   Output shape: {output_shape}")
                return True
            except Exception:
                continue
        raise RuntimeError('No model file could be loaded')
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def main():
    """Main startup function."""
    print("🚀 Starting Paddy Disease Detection API")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    print("\n📦 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    print("\n🔍 Checking model file...")
    if not check_model_file():
        sys.exit(1)
    
    print("\n🧪 Testing model...")
    if not test_model_loading():
        print("⚠️  Model loading failed, but continuing...")
    
    print("\n" + "=" * 50)
    print("✅ All checks passed! Starting Flask API...")
    print("🌐 API will be available at: http://localhost:5000")
    print("📱 Health check: http://localhost:5000/api/health")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Start the Flask app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
