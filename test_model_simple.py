#!/usr/bin/env python3
"""
Simple test script to verify the paddy disease detection model can be loaded.
"""

import os
import numpy as np

# Handle different TensorFlow/Keras versions
try:
    # Try tf.keras first (TensorFlow 2.x)
    from tensorflow import keras
    print(f"✅ Using TensorFlow with tf.keras")
except ImportError:
    try:
        # Fallback to standalone keras
        import keras
        print(f"✅ Using standalone Keras")
    except ImportError:
        print("❌ Neither tf.keras nor standalone keras is available")
        exit(1)

def test_model():
    """Test if the model can be loaded successfully."""
    try:
        print("🔍 Looking for model files...")
        
        # Check if model file exists
        model_path = 'neuralnetwork.keras'
        if os.path.exists(model_path):
            print(f"✅ Found model file: {model_path}")
        else:
            print(f"❌ Model file not found: {model_path}")
            return False
        
        print("📥 Loading model...")
        model = keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
        
        # Test basic model info
        print(f"📊 Model input shape: {model.input_shape}")
        print(f"📊 Model output shape: {model.output_shape}")
        
        # Test prediction with dummy data
        print("🧪 Testing prediction...")
        dummy_input = np.random.random((1, 224, 224, 3)).astype('float32')
        prediction = model.predict(dummy_input, verbose=0)
        print(f"✅ Prediction successful! Shape: {prediction.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Paddy Disease Detection Model")
    print("=" * 50)
    
    success = test_model()
    
    print("=" * 50)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Tests failed!")
