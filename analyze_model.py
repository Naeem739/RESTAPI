#!/usr/bin/env python3
"""
Model Analysis Script for Paddy Disease Detection
This script analyzes the loaded model to understand its architecture and input requirements.
"""

import tensorflow as tf
import numpy as np
from PIL import Image

def analyze_model_architecture():
    """Analyze the loaded model architecture."""
    try:
        print("🔍 Analyzing Paddy Disease Detection Model")
        print("=" * 60)
        
        # Load the model
        print("📥 Loading model...")
        model = tf.keras.models.load_model('model.h5')
        print("✅ Model loaded successfully!")
        
        # Basic model information
        print(f"\n📊 Model Summary:")
        print(f"   Input Shape: {model.input_shape}")
        print(f"   Output Shape: {model.output_shape}")
        print(f"   Number of Layers: {len(model.layers)}")
        print(f"   Total Parameters: {model.count_params():,}")
        
        # Analyze each layer
        print(f"\n🏗️  Layer Analysis:")
        for i, layer in enumerate(model.layers):
            layer_type = type(layer).__name__
            config = layer.get_config()
            
            print(f"\n   Layer {i+1}: {layer.name} ({layer_type})")
            print(f"      Input Shape: {layer.input_shape}")
            print(f"      Output Shape: {layer.output_shape}")
            
            # Special handling for specific layer types
            if layer_type == 'Flatten':
                print(f"      Flattened Size: {np.prod(layer.input_shape[1:])}")
            elif layer_type == 'Dense':
                print(f"      Units: {layer.units}")
                print(f"      Activation: {layer.activation.__name__ if hasattr(layer.activation, '__name__') else layer.activation}")
            elif layer_type == 'Conv2D':
                print(f"      Filters: {layer.filters}")
                print(f"      Kernel Size: {layer.kernel_size}")
                print(f"      Strides: {layer.strides}")
        
        # Test with different input sizes
        print(f"\n🧪 Testing Input Compatibility:")
        test_sizes = [(128, 128), (224, 224), (256, 256), (299, 299), (331, 331), (512, 512)]
        
        for size in test_sizes:
            try:
                # Create dummy image
                dummy_image = np.random.random((1, size[0], size[1], 3)).astype('float32')
                print(f"\n   Testing size {size}:")
                print(f"      Input shape: {dummy_image.shape}")
                
                # Try to get output shape without running prediction
                try:
                    # Use model's call method to get output shape
                    output_shape = model.compute_output_shape(dummy_image.shape)
                    print(f"      Expected output shape: {output_shape}")
                    print(f"      ✅ Compatible")
                except Exception as e:
                    print(f"      ❌ Incompatible: {str(e)[:100]}...")
                    
            except Exception as e:
                print(f"      ❌ Error testing size {size}: {e}")
        
        # Find the flatten layer and calculate expected input size
        print(f"\n🔍 Flatten Layer Analysis:")
        flatten_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Flatten)]
        
        if flatten_layers:
            for i, flatten_layer in enumerate(flatten_layers):
                print(f"   Flatten Layer {i+1}: {flatten_layer.name}")
                print(f"      Input Shape: {flatten_layer.input_shape}")
                print(f"      Output Shape: {flatten_layer.output_shape}")
                
                # Calculate the flattened size
                if flatten_layer.input_shape:
                    flattened_size = np.prod(flatten_layer.input_shape[1:])
                    print(f"      Flattened Size: {flattened_size}")
                    
                    # Check if this matches any dense layer input
                    for j, layer in enumerate(model.layers):
                        if isinstance(layer, tf.keras.layers.Dense) and j > i:
                            print(f"      → Feeds into Dense Layer {j+1}: {layer.name}")
                            print(f"         Expected Input: {layer.input_shape}")
                            break
        else:
            print("   No Flatten layers found")
        
        print(f"\n" + "=" * 60)
        print("🎯 Recommendations:")
        
        # Provide recommendations based on analysis
        if model.input_shape[1] != 128 or model.input_shape[2] != 128:
            print(f"   • Model expects input size: {model.input_shape[1]}x{model.input_shape[2]}")
            print(f"   • Training was done with 128x128")
            print(f"   • Update preprocessing to use {model.input_shape[1]}x{model.input_shape[2]}")
        else:
            print(f"   • ✅ Model expects 128x128 (matches training configuration)")
            print(f"   • Current preprocessing should work correctly")
        
        # Check for common issues
        if len(flatten_layers) > 0:
            flatten_layer = flatten_layers[0]
            if flatten_layer.input_shape:
                expected_size = np.prod(flatten_layer.input_shape[1:])
                print(f"   • Expected flattened input size: {expected_size}")
                print(f"   • Ensure image preprocessing produces this size")
        
        print(f"   • If issues persist, consider retraining the model")
        print(f"   • Check model architecture for layer compatibility")
        
        return model
        
    except Exception as e:
        print(f"❌ Error analyzing model: {e}")
        return None

def test_model_with_sample_input():
    """Test the model with a sample input to identify exact issues."""
    try:
        model = tf.keras.models.load_model('model.h5')
        print(f"\n🧪 Testing Model with Sample Input:")
        
        # Test with the expected input size
        expected_input_shape = model.input_shape
        print(f"   Expected input shape: {expected_input_shape}")
        
        # Create sample input
        sample_input = np.random.random(expected_input_shape).astype('float32')
        print(f"   Sample input shape: {sample_input.shape}")
        
        # Try prediction
        try:
            output = model.predict(sample_input, verbose=0)
            print(f"   ✅ Prediction successful!")
            print(f"   Output shape: {output.shape}")
            print(f"   Output sample: {output[0][:5]}")  # First 5 values
        except Exception as e:
            print(f"   ❌ Prediction failed: {e}")
            
    except Exception as e:
        print(f"❌ Error testing model: {e}")

if __name__ == "__main__":
    print("🚀 Paddy Disease Detection Model Analysis")
    print("=" * 60)
    
    # Analyze the model
    model = analyze_model_architecture()
    
    if model:
        # Test with sample input
        test_model_with_sample_input()
    
    print(f"\n✅ Analysis complete!")
