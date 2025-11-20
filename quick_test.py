#!/usr/bin/env python3
"""
Quick test to verify model loading works.
"""

import os
import numpy as np

# Handle different TensorFlow/Keras versions
try:
    from tensorflow import keras
    print(f"✅ Using TensorFlow with tf.keras")
except ImportError:
    try:
        import keras
        print(f"✅ Using standalone Keras")
    except ImportError:
        print("❌ Neither tf.keras nor standalone keras is available")
        exit(1)

def quick_test():
    """Quick test of model loading."""
    try:
        # Check if model file exists
        model_path = 'neuralnetwork.keras'
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
        
        print(f"✅ Found model file: {model_path}")
        
        # Load model
        print("📥 Loading model...")
        model = keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
        
        # Test prediction
        print("🧪 Testing prediction...")
        dummy_input = np.random.random((1, 224, 224, 3)).astype('float32')
        prediction = model.predict(dummy_input, verbose=0)
        print(f"✅ Prediction successful! Shape: {prediction.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Quick Model Test")
    print("=" * 30)
    
    success = quick_test()
    
    print("=" * 30)
    if success:
        print("✅ Model test passed!")
    else:
        print("❌ Model test failed!")
