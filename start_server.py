#!/usr/bin/env python3
"""
Startup script for the Paddy Disease Detection API server.
This script will test model loading and start the Flask server.
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking dependencies...")
    
    try:
        import flask
        print("✅ Flask is installed")
    except ImportError:
        print("❌ Flask is not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flask"])
    
    try:
        import tensorflow
        print(f"✅ TensorFlow is installed (version: {tensorflow.__version__})")
    except ImportError:
        print("❌ TensorFlow is not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow==2.15.0"])
    
    try:
        from tensorflow import keras
        print("✅ Keras is available")
    except ImportError:
        try:
            import keras
            print("✅ Standalone Keras is available")
        except ImportError:
            print("❌ Keras is not available. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "keras"])

def test_model_loading():
    """Test if the model can be loaded."""
    print("\n🧪 Testing model loading...")
    
    try:
        # Import the updated test script
        from test_model_simple import test_model
        success = test_model()
        
        if success:
            print("✅ Model loading test passed!")
            return True
        else:
            print("❌ Model loading test failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error during model test: {e}")
        return False

def start_server():
    """Start the Flask server."""
    print("\n🚀 Starting Flask server...")
    print("📡 Server will be available at: http://127.0.0.1:5000")
    print("📱 Flutter app should connect to: http://127.0.0.1:5000/api")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
    try:
        # Import and run the Flask app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

def main():
    """Main startup function."""
    print("🌾 Paddy Disease Detection API Server")
    print("=" * 50)
    
    # Check dependencies
    check_dependencies()
    
    # Test model loading
    if not test_model_loading():
        print("\n❌ Cannot start server - model loading failed!")
        print("💡 Please check:")
        print("   1. The neuralnetwork.keras file exists in the flask_api directory")
        print("   2. TensorFlow/Keras is properly installed")
        print("   3. The model file is not corrupted")
        return
    
    # Start server
    start_server()

if __name__ == "__main__":
    main()
