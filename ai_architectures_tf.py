"""
TensorFlow/Keras CNN Model Implementation for the existing architecture
"""

import os
import numpy as np
import joblib
from typing import Dict, List, Tuple, Any, Optional
from ai_architectures import ModelInterface, DataPreprocessorInterface

# Try to import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. CNN model will not work.")


class CNNPreprocessor(DataPreprocessorInterface):
    """
    Preprocessor specifically for the CNN model (100x100 images to match trained model)
    """
    
    def __init__(self, target_size: Tuple[int, int] = (100, 100)):
        self.target_size = target_size
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        # Similar logic to StandardImagePreprocessor but ensuring proper normalization for CNN
        if image is None or image.size == 0:
            raise ValueError("Empty or invalid image")
        
        current_image_data: np.ndarray

        if image.ndim == 3 and image.shape[2] == 4:  # RGBA
            gray_image = np.dot(image[:, :, :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            current_image_data = 255 - gray_image
        elif image.ndim == 3 and image.shape[2] == 3:  # RGB
            gray_image = np.dot(image[:, :, :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            current_image_data = 255 - gray_image
        elif image.ndim == 2:  # Already grayscale
            if np.mean(image) > 128: 
                current_image_data = 255 - image.astype(np.uint8)
            else:
                current_image_data = image.astype(np.uint8)
        else:
            raise ValueError(f"Unsupported image format: shape {image.shape}, ndim {image.ndim}")
        
        # Resize to 100x100 for CNN
        from skimage import transform
        image_resized = transform.resize(
            current_image_data, self.target_size,
            anti_aliasing=True, preserve_range=True
        )
        image_resized = np.clip(image_resized, 0, 255).astype(np.uint8)
        
        # Normalize to [0, 1] for TensorFlow
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Add channel dimension if needed
        if image_normalized.ndim == 2:
            image_normalized = np.expand_dims(image_normalized, axis=-1)
        
        return image_normalized
    
    def preprocess_dataset(self, images: List[np.ndarray]) -> np.ndarray:
        if not images:
            raise ValueError("Empty image list")
        
        processed_images = []
        for i, img in enumerate(images):
            try:
                processed_img = self.preprocess_image(img)
                processed_images.append(processed_img)
            except Exception as e:
                print(f"[CNN_PREPROCESSOR] Error processing image {i}: {e}")
                continue
        
        if not processed_images:
            raise ValueError("No images were successfully processed")
        
        result = np.array(processed_images)
        print(f"[CNN_PREPROCESSOR] Processed {len(processed_images)} images to shape {result.shape}")
        
        return result


class TensorFlowCNNModel(ModelInterface):
    """
    Real CNN implementation using TensorFlow/Keras
    """
    
    def __init__(self, symbols_display: Dict[str, str], preprocessor: DataPreprocessorInterface = None):
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for CNN model but not installed")
        
        self.model = None
        self.symbols_display = symbols_display
        self.preprocessor = preprocessor or CNNPreprocessor(target_size=(100, 100))
        self.is_trained = False
        self.training_accuracy = 0.0
        self.classes_ = ['alpha', 'beta', 'epsilon']  # Default classes
        self.model_path_h5 = None
        
    def _create_model(self, input_shape=(64, 64, 1), num_classes=3):
        """Create the CNN architecture similar to notebook_cnn.py"""
        model = keras.Sequential([
            # First convolutional block
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),

            # Second convolutional block
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),

            # Third convolutional block
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.25),

            # Flatten and dense layers
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X: List[np.ndarray], y: np.ndarray) -> float:
        """Train would not be used for pre-trained model, but implemented for interface"""
        print("[TF_CNN] This model is designed to load pre-trained weights.")
        print("[TF_CNN] Training from scratch is not recommended. Use load_model() instead.")
        return 0.0
    
    def predict(self, X: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be loaded before prediction")
        
        # Preprocess single image
        X_processed = self.preprocessor.preprocess_image(X)
        
        # Add batch dimension
        X_batch = np.expand_dims(X_processed, axis=0)
        
        # Predict
        predictions = self.model.predict(X_batch, verbose=0)[0]
        predicted_class_idx = np.argmax(predictions)
        
        # Map to class name
        predicted_class = self.classes_[predicted_class_idx]
        confidence = float(predictions[predicted_class_idx])
        
        # Build all predictions
        all_predictions = {}
        for i, class_name in enumerate(self.classes_):
            all_predictions[class_name] = {
                'symbol': self.symbols_display.get(class_name, class_name),
                'probability': float(predictions[i])
            }
        
        predicted_symbol = self.symbols_display.get(predicted_class, predicted_class)
        
        return predicted_symbol, confidence, all_predictions
    
    def save_model(self, path: str) -> None:
        """Save model in both .h5 and .pkl format for compatibility"""
        if not self.is_trained or self.model is None:
            raise ValueError("Cannot save untrained model")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the keras model as .h5
        h5_path = path.replace('.pkl', '.h5')
        self.model.save(h5_path)
        
        # Save metadata as .pkl
        model_data = {
            'model_path_h5': h5_path,
            'symbols_display': self.symbols_display,
            'training_accuracy': self.training_accuracy,
            'classes': self.classes_,
            'preprocessor_config': {'target_size': self.preprocessor.target_size},
            'model_type': 'tensorflow_cnn'
        }
        joblib.dump(model_data, path)
        print(f"[TF_CNN] Model saved to {h5_path} and metadata to {path}")
    
    def load_model(self, path: str) -> bool:
        """Load model from .h5 file or from .pkl metadata"""
        try:
            if path.endswith('.h5'):
                # Direct H5 load
                self.model = keras.models.load_model(path)
                self.model_path_h5 = path
                self.is_trained = True
                # Try to infer other properties
                self.training_accuracy = 0.9  # Default assumption
                print(f"[TF_CNN] Loaded model directly from {path}")
                return True
            
            elif path.endswith('.pkl'):
                # Load from metadata
                if not os.path.exists(path):
                    # Try to load the .h5 directly if .pkl doesn't exist
                    h5_path = path.replace('.pkl', '.h5')
                    if os.path.exists(h5_path):
                        return self.load_model(h5_path)
                    return False
                
                model_data = joblib.load(path)
                h5_path = model_data.get('model_path_h5', path.replace('.pkl', '.h5'))
                
                # Load the actual model
                if os.path.exists(h5_path):
                    self.model = keras.models.load_model(h5_path)
                else:
                    print(f"[TF_CNN] H5 file not found at {h5_path}")
                    return False
                
                # Load metadata
                self.symbols_display = model_data.get('symbols_display', self.symbols_display)
                self.training_accuracy = model_data.get('training_accuracy', 0.9)
                self.classes_ = model_data.get('classes', self.classes_)
                self.model_path_h5 = h5_path
                self.is_trained = True
                
                print(f"[TF_CNN] Loaded model from {h5_path} with metadata from {path}")
                return True
            
            else:
                # Try .h5 file with same name
                h5_path = os.path.join(os.path.dirname(path), 'mejor_modelo.h5')
                if os.path.exists(h5_path):
                    return self.load_model(h5_path)
                
            return False
            
        except Exception as e:
            print(f"[TF_CNN] Error loading model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        model_summary = "Not loaded"
        total_params = 0
        
        if self.model is not None:
            # Get model summary
            total_params = self.model.count_params()
            model_summary = f"{len(self.model.layers)} layers, {total_params:,} parameters"
        
        return {
            'type': 'TensorFlow CNN',
            'architecture': 'Convolutional Neural Network',
            'is_trained': self.is_trained,
            'training_accuracy': self.training_accuracy,
            'classes': self.classes_,
            'preprocessing': 'CNNPreprocessor (100x100)',
            'framework': 'TensorFlow/Keras',
            'model_summary': model_summary,
            'total_params': total_params,
            'input_shape': '(100, 100, 1)'
        }


# Wrapper class to make it work seamlessly with the existing system
class TensorFlowCNNWrapper(ModelInterface):
    """
    Wrapper that can load from .h5 and save as .pkl for compatibility
    """
    
    def __init__(self, symbols_display: Dict[str, str], preprocessor: DataPreprocessorInterface = None):
        self.tf_model = TensorFlowCNNModel(symbols_display, preprocessor)
        self.symbols_display = symbols_display
        self.preprocessor = preprocessor or CNNPreprocessor(target_size=(100, 100))
        
    def train(self, X: List[np.ndarray], y: np.ndarray) -> float:
        return self.tf_model.train(X, y)
    
    def predict(self, X: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        return self.tf_model.predict(X)
    
    def save_model(self, path: str) -> None:
        self.tf_model.save_model(path)
    
    def load_model(self, path: str) -> bool:
        # Special handling for existing .h5 file
        if 'modelo_cnn' in path and path.endswith('.pkl'):
            # Check if mejor_modelo.h5 exists
            h5_path = os.path.join(os.path.dirname(path), 'mejor_modelo.h5')
            if os.path.exists(h5_path):
                print(f"[CNN_WRAPPER] Loading pre-trained model from {h5_path}")
                return self.tf_model.load_model(h5_path)
        
        return self.tf_model.load_model(path)
    
    def get_model_info(self) -> Dict[str, Any]:
        return self.tf_model.get_model_info() 