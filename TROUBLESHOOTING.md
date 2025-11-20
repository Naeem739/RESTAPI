# Troubleshooting Guide - Paddy Disease Detection API

## Issue: "Model not Found" Error

### Problem Description
When clicking the "Detect Disease" button in the Flutter app, you get a "Model not Found" error. This happens because the TensorFlow/Keras model cannot be loaded properly.

### Root Cause
The error `module 'tensorflow' has no attribute 'keras'` indicates that you're using an incompatible version of TensorFlow where Keras is not available as `tf.keras`.

### Solution Steps

#### Step 1: Update TensorFlow Version
The recommended TensorFlow version for this project is 2.15.0. Run these commands:

```bash
cd flask_api
pip uninstall tensorflow
pip install tensorflow==2.15.0
```

#### Step 2: Test Model Loading
Run the updated test script to verify the model loads correctly:

```bash
python test_model_simple.py
```

You should see output like:
```
✅ Using TensorFlow with tf.keras
🔍 Looking for model files...
✅ Found model file: neuralnetwork.keras
📥 Loading model...
✅ Model loaded successfully!
📊 Model input shape: (None, 224, 224, 3)
📊 Model output shape: (None, 10)
🧪 Testing prediction...
✅ Prediction successful! Shape: (1, 10)
✅ All tests passed!
```

#### Step 3: Start the Server
Use the new startup script that includes dependency checking:

```bash
python start_server.py
```

Or manually start the server:

```bash
python app.py
```

#### Step 4: Test the API
Once the server is running, test the API endpoint:

```bash
curl http://127.0.0.1:5000/api/health
```

You should get:
```json
{"status": "healthy", "message": "Paddy Disease Detector API is running"}
```

### Alternative Solutions

#### If TensorFlow 2.15.0 doesn't work:
Try installing a different version:

```bash
pip install tensorflow==2.14.0
```

#### If you still get Keras import errors:
Install standalone Keras:

```bash
pip install keras
```

#### If the model file is corrupted:
1. Check if `neuralnetwork.keras` exists in the `flask_api` directory
2. Verify the file size (should be around 4.6MB)
3. If corrupted, you may need to retrain or download the model again

### Verification Steps

1. **Check TensorFlow Version:**
   ```bash
   python -c "import tensorflow as tf; print(tf.__version__)"
   ```

2. **Check Keras Availability:**
   ```bash
   python -c "from tensorflow import keras; print('Keras available')"
   ```

3. **Test Model Loading:**
   ```bash
   python quick_test.py
   ```

4. **Start Server and Test:**
   ```bash
   python start_server.py
   ```

### Common Error Messages and Solutions

| Error Message | Solution |
|---------------|----------|
| `module 'tensorflow' has no attribute 'keras'` | Update TensorFlow to version 2.15.0 |
| `Model not available` | Check if `neuralnetwork.keras` file exists |
| `Cannot connect to server` | Start the Flask server with `python app.py` |
| `Model output mismatch` | Check if DISEASE_CLASSES matches model output |

### File Structure
Ensure your `flask_api` directory contains:
```
flask_api/
├── app.py                    # Main Flask application
├── neuralnetwork.keras       # Model file (4.6MB)
├── requirements.txt          # Dependencies
├── test_model_simple.py      # Model test script
├── start_server.py           # Startup script
└── quick_test.py            # Quick test script
```

### Support
If you continue to have issues:
1. Check the console output for detailed error messages
2. Verify all dependencies are installed correctly
3. Ensure the model file is not corrupted
4. Try running the test scripts to isolate the issue
