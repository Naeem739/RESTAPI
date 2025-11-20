#!/usr/bin/env python3
"""
Test script to verify the paddy disease detection model can be loaded and used.
"""

import numpy as np
from PIL import Image
import tensorflow as tf
import os # Added missing import

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
        exit(1)

def test_model_loading():
    """Test if the model can be loaded successfully."""
    try:
        print("Attempting to load the model...")
        
        # Try different model file names and formats
        model_paths = [
            'neuralnetwork.keras',
            'model.h5',
            './neuralnetwork.keras',
            './model.h5'
        ]
        
        model = None
        for path in model_paths:
            try:
                if os.path.exists(path):
                    print(f"Trying to load model from: {path}")
                    model = keras.models.load_model(path)
                    print(f"✅ Model loaded successfully from {path}!")
                    break
            except Exception as e:
                print(f"Failed to load from {path}: {e}")
                continue
        
        if model is None:
            print("❌ Could not load model from any path")
            return None
        
        # Get model summary
        print("\nModel Summary:")
        model.summary()
        
        # Get input shape
        input_shape = model.input_shape
        print(f"\nInput shape: {input_shape}")
        
        # Get output shape
        output_shape = model.output_shape
        print(f"Output shape: {output_shape}")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def test_prediction(model):
    """Test if the model can make predictions."""
    if model is None:
        print("Cannot test prediction - model not loaded")
        return
    
    try:
        print("\nTesting prediction with dummy data...")
        
        # Create dummy image data (224x224 RGB image)
        dummy_image = np.random.random((1, 224, 224, 3)).astype('float32')
        
        # Make prediction
        predictions = model.predict(dummy_image)
        
        print("✅ Prediction successful!")
        print(f"Prediction shape: {predictions.shape}")
        print(f"Sample predictions: {predictions[0][:5]}")  # Show first 5 values
        
        # Get predicted class
        predicted_class_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_index])
        
        print(f"Predicted class index: {predicted_class_index}")
        print(f"Confidence: {confidence:.4f}")
        
    except Exception as e:
        print(f"❌ Error making prediction: {e}")

def main():
    """Main test function."""
    print("🧪 Testing Paddy Disease Detection Model")
    print("=" * 50)
    
    # Test model loading
    model = test_model_loading()
    
    # Test prediction
    test_prediction(model)
    
    print("\n" + "=" * 50)
    print("Test completed!")

if __name__ == "__main__":
    main()
