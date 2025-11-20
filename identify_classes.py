#!/usr/bin/env python3
"""
Script to identify the actual classes in your paddy disease detection model.
This will help you understand what your 3-class model is actually detecting.
"""

import tensorflow as tf
import numpy as np

def identify_model_classes():
    """Identify the actual classes in your model."""
    try:
        print("🔍 Identifying Model Classes")
        print("=" * 50)
        
        # Load the model
        print("📥 Loading model...")
        model = tf.keras.models.load_model('model.h5')
        print("✅ Model loaded successfully!")
        
        # Basic model information
        print(f"\n📊 Model Summary:")
        print(f"   Input Shape: {model.input_shape}")
        print(f"   Output Shape: {model.output_shape}")
        print(f"   Number of Classes: {model.output_shape[1]}")
        
        # Test with sample input
        print(f"\n🧪 Testing Model Output:")
        
        # Create sample input (128x128 as specified)
        sample_input = np.random.random((1, 128, 128, 3)).astype('float32')
        print(f"   Input shape: {sample_input.shape}")
        
        # Get prediction
        predictions = model.predict(sample_input, verbose=0)
        print(f"   Output shape: {predictions.shape}")
        print(f"   Raw predictions: {predictions[0]}")
        
        # Analyze the output
        print(f"\n🔍 Class Analysis:")
        for i, prob in enumerate(predictions[0]):
            print(f"   Class {i}: {prob:.6f} ({prob*100:.2f}%)")
        
        # Find the predicted class
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        print(f"\n🎯 Prediction Result:")
        print(f"   Predicted Class: {predicted_class}")
        print(f"   Confidence: {confidence:.6f} ({confidence*100:.2f}%)")
        
        # Check if this is a binary or multi-class model
        if model.output_shape[1] == 2:
            print(f"\n📋 Model Type: Binary Classification")
            print(f"   Class 0: Likely 'Healthy' or 'No Disease'")
            print(f"   Class 1: Likely 'Disease' or 'Infected'")
        elif model.output_shape[1] == 3:
            print(f"\n📋 Model Type: 3-Class Classification")
            print(f"   Class 0: Likely 'Healthy'")
            print(f"   Class 1: Likely 'Disease Type 1' (e.g., Bacterial Blight)")
            print(f"   Class 2: Likely 'Disease Type 2' (e.g., Leaf Blast)")
        else:
            print(f"\n📋 Model Type: {model.output_shape[1]}-Class Classification")
        
        # Provide recommendations
        print(f"\n🎯 Recommendations:")
        print(f"   • Your model has {model.output_shape[1]} output classes")
        print(f"   • Update DISEASE_CLASSES in app.py to match this number")
        print(f"   • Consider what each class represents based on your training data")
        print(f"   • If you need 12 classes, retrain the model with more disease types")
        
        return model
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_with_real_image():
    """Test the model with a real image to see actual predictions."""
    try:
        print(f"\n🖼️  Testing with Real Image:")
        print("=" * 50)
        
        # Import the preprocessing function
        from app import preprocess_image
        
        # Create a simple test image
        from PIL import Image
        from io import BytesIO
        
        # Create a green test image (simulating a healthy leaf)
        test_img = Image.new('RGB', (256, 256), color='green')
        img_buffer = BytesIO()
        test_img.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        
        # Preprocess the image
        processed = preprocess_image(img_buffer)
        if processed is not None:
            print(f"✅ Image preprocessed: {processed.shape}")
            
            # Load model and predict
            model = tf.keras.models.load_model('model.h5')
            predictions = model.predict(processed, verbose=0)
            
            print(f"📊 Predictions for green image:")
            for i, prob in enumerate(predictions[0]):
                print(f"   Class {i}: {prob:.6f} ({prob*100:.2f}%)")
            
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            print(f"🎯 Predicted: Class {predicted_class} with {confidence*100:.2f}% confidence")
            
        else:
            print("❌ Image preprocessing failed")
            
    except Exception as e:
        print(f"❌ Error testing with real image: {e}")

if __name__ == "__main__":
    print("🚀 Paddy Disease Model Class Identification")
    print("=" * 60)
    
    # Identify the classes
    model = identify_model_classes()
    
    if model:
        # Test with a real image
        test_with_real_image()
    
    print(f"\n✅ Analysis complete!")
    print(f"💡 Update your DISEASE_CLASSES list based on the results above")
