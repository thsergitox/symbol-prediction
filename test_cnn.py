#!/usr/bin/env python3
"""
Script de prueba para verificar la implementación CNN con TensorFlow/Keras
"""

import sys
import os

# Add the current directory to path to import local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    print("🔍 Testing TensorFlow import...")
    import tensorflow as tf
    print(f"✅ TensorFlow version: {tf.__version__}")
    print(f"✅ GPU available: {tf.config.list_physical_devices('GPU')}")
    
    print("\n🔍 Testing CNN architecture import...")
    from ai_architectures import ConvolutionalNeuralNetwork, CNNImagePreprocessor
    print("✅ CNN classes imported successfully")
    
    print("\n🔍 Testing CNN initialization...")
    symbols_display = {'alpha': 'α', 'beta': 'β', 'epsilon': 'ε'}
    preprocessor = CNNImagePreprocessor(target_size=(64, 64))
    cnn = ConvolutionalNeuralNetwork(symbols_display, preprocessor)
    print("✅ CNN initialized successfully")
    
    print(f"\n📊 CNN Model Info:")
    info = cnn.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n🔍 Testing model creation...")
    model = cnn._create_cnn_model('basic')
    print("✅ CNN model architecture created successfully")
    print(f"  Total parameters: {model.count_params():,}")
    
    # Test with dummy data
    print("\n🔍 Testing with dummy data...")
    import numpy as np
    
    # Create dummy image data
    dummy_images = [np.random.randint(0, 255, (100, 100), dtype=np.uint8) for _ in range(6)]
    dummy_labels = ['alpha', 'beta', 'epsilon'] * 2
    
    print(f"  Created {len(dummy_images)} dummy images")
    print(f"  Labels: {dummy_labels}")
    
    # Test preprocessing
    print("\n🔍 Testing preprocessing...")
    processed = preprocessor.preprocess_dataset(dummy_images)
    print(f"✅ Preprocessing successful - shape: {processed.shape}, dtype: {processed.dtype}")
    print(f"  Data range: [{processed.min():.3f}, {processed.max():.3f}]")
    
    # Test training (with very few epochs for quick test)
    print("\n🔍 Testing training (quick test with 2 epochs)...")
    cnn.epochs = 2
    cnn.batch_size = 4
    
    accuracy = cnn.train(dummy_images, np.array(dummy_labels))
    print(f"✅ Training completed with accuracy: {accuracy:.4f}")
    
    # Test prediction
    print("\n🔍 Testing prediction...")
    test_image = dummy_images[0]
    prediction, confidence, all_predictions = cnn.predict(test_image)
    print(f"✅ Prediction: {prediction}, Confidence: {confidence:.4f}")
    print(f"  All predictions: {list(all_predictions.keys())}")
    
    print("\n🎉 All tests passed! CNN implementation is working correctly.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required dependencies:")
    print("  pip install tensorflow keras opencv-python matplotlib seaborn pandas")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    print(f"Full traceback:\n{traceback.format_exc()}")
    sys.exit(1) 