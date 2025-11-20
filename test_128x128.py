#!/usr/bin/env python3
"""
Quick test script to verify 128x128 image processing works with the model.
"""

import tensorflow as tf
import numpy as np
from PIL import Image

def test_128x128_input():
    """Test if the model accepts 128x128 input."""
    try:
        print("🧪 Testing 128x128 Input Compatibility")
        print("=" * 50)
        
        # Load the model
        print("📥 Loading model...")
        model = tf.keras.models.load_model('model.h5')
        print("✅ Model loaded successfully!")
        
        # Check model input shape
        print(f"🔍 Model input shape: {model.input_shape}")
        
        # Create 128x128 test image
        print("\n🖼️  Creating 128x128 test image...")
        test_image = np.random.random((1, 128, 128, 3)).astype('float32')
        print(f"   Test image shape: {test_image.shape}")
        
        # Try prediction
        print("\n🚀 Testing prediction...")
        try:
            output = model.predict(test_image, verbose=0)
            print("✅ Prediction successful!")
            print(f"   Output shape: {output.shape}")
            print(f"   Output sample: {output[0][:5]}")
            
            # Check if output matches expected 12 classes
            if output.shape[1] == 12:
                print("✅ Output has 12 classes (matches disease classes)")
            else:
                print(f"⚠️  Expected 12 classes, got {output.shape[1]}")
                
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_image_preprocessing():
    """Test the image preprocessing function."""
    try:
        print("\n🔄 Testing Image Preprocessing")
        print("=" * 50)
        
        # Import the preprocessing function
        from app import preprocess_image
        
        # Create a dummy image file (simulate uploaded file)
        from io import BytesIO
        
        # Create a test image
        test_img = Image.new('RGB', (256, 256), color='green')
        img_buffer = BytesIO()
        test_img.save(img_buffer, format='JPEG')
        img_buffer.seek(0)
        
        # Test preprocessing
        print("📸 Testing with 256x256 input image...")
        processed = preprocess_image(img_buffer)
        
        if processed is not None:
            print(f"✅ Preprocessing successful!")
            print(f"   Output shape: {processed.shape}")
            print(f"   Data type: {processed.dtype}")
            print(f"   Value range: {processed.min():.3f} to {processed.max():.3f}")
            return True
        else:
            print("❌ Preprocessing failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing preprocessing: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing 128x128 Model Compatibility")
    print("=" * 60)
    
    # Test 1: Direct model input
    success1 = test_128x128_input()
    
    # Test 2: Image preprocessing
    success2 = test_image_preprocessing()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 All tests passed! Your model should work with 128x128 images.")
        print("📱 Try uploading an image in the Flutter app now.")
    else:
        print("⚠️  Some tests failed. Check the error messages above.")
    
    print("=" * 60)



