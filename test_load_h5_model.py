import tensorflow as tf
import numpy as np

print("Loading the pre-trained CNN model...")
try:
    model = tf.keras.models.load_model('model/mejor_modelo.h5')
    print("✅ Model loaded successfully!")
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    # Get model details
    print(f"\nInput shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Number of layers: {len(model.layers)}")
    print(f"Total parameters: {model.count_params():,}")
    
    # Test with a dummy input
    dummy_input = np.random.rand(1, 100, 100, 1).astype(np.float32)
    predictions = model.predict(dummy_input, verbose=0)
    print(f"\nTest prediction shape: {predictions.shape}")
    print(f"Test prediction: {predictions[0]}")
    print(f"Sum of probabilities: {np.sum(predictions[0]):.4f}")
    
    # Test with actual data
    print("\n\nTesting with actual dataset image...")
    X = np.load('dataset/X.npy')
    test_idx = 0
    test_image = X[test_idx]
    if test_image.ndim == 1:
        side = int(np.sqrt(test_image.shape[0]))
        test_image = test_image.reshape(side, side)
    
    # Normalize and add dimensions
    test_image = test_image.astype(np.float32)
    if test_image.max() > 1:
        test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=-1)
    test_image = np.expand_dims(test_image, axis=0)
    
    predictions = model.predict(test_image, verbose=0)
    print(f"Predictions for dataset image: {predictions[0]}")
    print(f"Predicted class: {np.argmax(predictions[0])} (0=alpha, 1=beta, 2=epsilon)")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    import traceback
    traceback.print_exc() 