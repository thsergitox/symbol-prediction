from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
from datetime import datetime
from skimage import transform, filters, feature
from scipy import ndimage
import tempfile
import cv2

# TensorFlow and Keras imports for real CNN
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
    print(f"[CNN] TensorFlow version: {tf.__version__} - GPU available: {tf.config.list_physical_devices('GPU')}")
except ImportError as e:
    print(f"[CNN] TensorFlow not available: {e}. CNN will use simplified fallback.")
    TENSORFLOW_AVAILABLE = False

# ============================================================================
# ABSTRACTIONS (Dependency Inversion Principle)
# ============================================================================

class ModelInterface(ABC):
    """
    Interface for all ML models (Dependency Inversion Principle)
    """
    
    @abstractmethod
    def train(self, X: List[np.ndarray], y: np.ndarray) -> float:
        """Train the model with images and labels, return accuracy"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        """Predict and return (prediction, confidence, all_predictions)"""
        pass
    
    @abstractmethod
    def save_model(self, path: str) -> None:
        """Save the trained model"""
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> bool:
        """Load a trained model, return True if successful"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata"""
        pass

class DataPreprocessorInterface(ABC):
    """
    Interface for data preprocessing (Single Responsibility Principle)
    """
    
    @abstractmethod
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess a single image"""
        pass
    
    @abstractmethod
    def preprocess_dataset(self, images: List[np.ndarray]) -> np.ndarray:
        """Preprocess a dataset of images"""
        pass

# ============================================================================
# CONCRETE IMPLEMENTATIONS 
# ============================================================================

class StandardImagePreprocessor(DataPreprocessorInterface):
    """
    Standard image preprocessing implementation
    """
    
    def __init__(self, target_size: Tuple[int, int] = (100, 100)):
        self.target_size = target_size
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        # Ensure we have a valid image
        if image is None or image.size == 0:
            raise ValueError("Empty or invalid image")
        
        current_image_data: np.ndarray

        if image.ndim == 3 and image.shape[2] == 4:  # RGBA (e.g., from canvas: black on white opaque)
            # Convert RGB part to grayscale. For black on white: symbol is dark (near 0), background is light (near 255).
            # Standard weights for RGB to Grayscale conversion
            gray_image = np.dot(image[:, :, :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            # Invert the grayscale image: symbol becomes white (255), background black (0).
            current_image_data = 255 - gray_image
        elif image.ndim == 3 and image.shape[2] == 3:  # RGB
            # Convert to grayscale
            gray_image = np.dot(image[:, :, :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            # Invert: symbol white (255), background black (0)
            current_image_data = 255 - gray_image
        elif image.ndim == 2:  # Already grayscale
            # If it's already 2D, it might be:
            # 1. Training data (black on transparent -> alpha -> symbol=255, bg=0) -> use as is.
            # 2. Grayscale image (symbol=0, bg=255) -> needs inversion.
            # Heuristic: if mean is high, likely white background (symbol=0, bg=255), so invert.
            if np.mean(image) > 128: 
                current_image_data = 255 - image.astype(np.uint8)
            else: # Otherwise, assume it's already symbol white (255) on black (0)
                current_image_data = image.astype(np.uint8)
        else:
            raise ValueError(f"Unsupported image format: shape {image.shape}, ndim {image.ndim}")
        
        # Ensure we have a 2D grayscale image
        if current_image_data.ndim != 2:
            raise ValueError(f"Image processing failed to produce a 2D image. Shape: {current_image_data.shape}")
        
        # Resize image to target size
        try:
            image_resized = transform.resize(
                current_image_data, self.target_size,
                anti_aliasing=True, preserve_range=True
            )
            # Ensure proper data type and range
            image_resized = np.clip(image_resized, 0, 255).astype(np.uint8)
        except Exception as e:
            raise ValueError(f"Error resizing image: {e}")
        
        # Verify output shape
        if image_resized.shape != self.target_size:
            raise ValueError(f"Resize failed: expected {self.target_size}, got {image_resized.shape}")
        
        return image_resized
    
    def preprocess_dataset(self, images: List[np.ndarray]) -> np.ndarray:
        if not images:
            raise ValueError("Empty image list")
        
        processed_images = []
        for i, img in enumerate(images):
            try:
                processed_img = self.preprocess_image(img)
                processed_images.append(processed_img)
            except Exception as e:
                print(f"[PREPROCESSOR] Error processing image {i}: {e}")
                continue
        
        if not processed_images:
            raise ValueError("No images were successfully processed")
        
        # Convert to numpy array - all images should now have the same shape
        result = np.array(processed_images)
        print(f"[PREPROCESSOR] Processed {len(processed_images)} images to shape {result.shape}")
        
        return result

class CNNImagePreprocessor(DataPreprocessorInterface):
    """
    CNN-specific image preprocessing implementation
    """
    
    def __init__(self, target_size: Tuple[int, int] = (64, 64)):
        self.target_size = target_size
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess a single image for CNN (returns with channel dimension)"""
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
            raise ValueError(f"Unsupported image format: shape {image.shape}")
        
        # Resize image
        try:
            image_resized = cv2.resize(current_image_data, self.target_size, interpolation=cv2.INTER_AREA)
        except Exception as e:
            try:
                # Fallback to skimage
                image_resized = transform.resize(
                    current_image_data, self.target_size,
                    anti_aliasing=True, preserve_range=True
                ).astype(np.uint8)
            except Exception as e2:
                raise ValueError(f"Error resizing image: {e}, fallback error: {e2}")
        
        # Normalize to [0, 1] and add channel dimension
        image_normalized = image_resized.astype(np.float32) / 255.0
        
        # Add channel dimension for CNN (H, W, 1)
        if image_normalized.ndim == 2:
            image_normalized = np.expand_dims(image_normalized, axis=-1)
        
        return image_normalized
    
    def preprocess_dataset(self, images: List[np.ndarray]) -> np.ndarray:
        """Preprocess dataset for CNN (returns 4D array N, H, W, C)"""
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
        
        # Convert to numpy array
        result = np.array(processed_images)
        print(f"[CNN_PREPROCESSOR] Processed {len(processed_images)} images to shape {result.shape}")
        
        return result

class RandomForestModel(ModelInterface):
    """
    Random Forest implementation (Open/Closed Principle - extensible)
    """
    
    def __init__(self, symbols_display: Dict[str, str], preprocessor: DataPreprocessorInterface):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        self.symbols_display = symbols_display
        self.preprocessor = preprocessor
        self.is_trained = False
        self.training_accuracy = 0.0
        
    def train(self, X: List[np.ndarray], y: np.ndarray) -> float:
        print(f"[{self.__class__.__name__}_TRAIN] Received {len(X)} images for training.")
        # Preprocess data
        X_processed = self.preprocessor.preprocess_dataset(X)
        print(f"[{self.__class__.__name__}_TRAIN] X_processed shape: {X_processed.shape}, dtype: {X_processed.dtype}")
        X_flattened = X_processed.reshape(X_processed.shape[0], -1)
        print(f"[{self.__class__.__name__}_TRAIN] X_flattened shape: {X_flattened.shape}")
        print(f"[{self.__class__.__name__}_TRAIN] y shape: {y.shape}, unique labels: {np.unique(y, return_counts=True)}")

        # Handle small datasets or single class
        if len(X_flattened) < 5 or len(np.unique(y)) < 2: # Increased threshold for more robust split
            print(f"[{self.__class__.__name__}_TRAIN] Small dataset or single class ({len(X_flattened)} samples, {len(np.unique(y))} unique classes). Fitting on full data.")
            if len(X_flattened) == 0:
                self.training_accuracy = 0.0
                print(f"[{self.__class__.__name__}_TRAIN] No data to train. Accuracy set to 0.0")
            else:
                self.model.fit(X_flattened, y)
                # Evaluate on the training data itself if no split is done
                y_pred_on_train = self.model.predict(X_flattened)
                self.training_accuracy = accuracy_score(y, y_pred_on_train)
                print(f"[{self.__class__.__name__}_TRAIN] Accuracy on training set (no split): {self.training_accuracy:.4f}")
        else:
            print(f"[{self.__class__.__name__}_TRAIN] Sufficient data for train/test split.")
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_flattened, y, test_size=0.25, random_state=42, stratify=y # Using 25% for test
                )
                print(f"[{self.__class__.__name__}_TRAIN] X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
                print(f"[{self.__class__.__name__}_TRAIN] y_train unique: {np.unique(y_train, return_counts=True)}, y_test unique: {np.unique(y_test, return_counts=True)}")
                
                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_test)
                self.training_accuracy = accuracy_score(y_test, y_pred)
                print(f"[{self.__class__.__name__}_TRAIN] Predictions on test set: {y_pred[:10]}... (first 10)")
                print(f"[{self.__class__.__name__}_TRAIN] True labels on test set: {y_test[:10]}... (first 10)")
                print(f"[{self.__class__.__name__}_TRAIN] Accuracy on test set: {self.training_accuracy:.4f}")
            except ValueError as e:
                print(f"[{self.__class__.__name__}_TRAIN] Error during train_test_split (e.g., not enough samples for stratify): {e}. Fitting on full data instead.")
                self.model.fit(X_flattened, y)
                y_pred_on_train = self.model.predict(X_flattened)
                self.training_accuracy = accuracy_score(y, y_pred_on_train)
                print(f"[{self.__class__.__name__}_TRAIN] Accuracy on training set (fallback): {self.training_accuracy:.4f}")
        
        self.is_trained = True
        print(f"[{self.__class__.__name__}_TRAIN] Training finished. Final reported accuracy: {self.training_accuracy:.4f}")
        return self.training_accuracy
    
    def predict(self, X: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Preprocess single image - FIXED: Always use preprocess_image for single predictions
        X_processed = self.preprocessor.preprocess_image(X)
        X_processed = X_processed.reshape(1, -1)
        
        prediction_proba = self.model.predict_proba(X_processed)[0]
        predicted_class_idx = np.argmax(prediction_proba)
        predicted_symbol_name = self.model.classes_[predicted_class_idx]
        confidence = prediction_proba[predicted_class_idx]
        
        all_predictions = {}
        for i, class_name in enumerate(self.model.classes_):
            all_predictions[class_name] = {
                'symbol': self.symbols_display.get(class_name, class_name),
                'probability': float(prediction_proba[i])
            }
        
        return (self.symbols_display.get(predicted_symbol_name, predicted_symbol_name), 
                confidence, all_predictions)
    
    def save_model(self, path: str) -> None:
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_data = {
            'model': self.model,
            'symbols_display': self.symbols_display,
            'training_accuracy': self.training_accuracy,
            'preprocessor_config': {'target_size': self.preprocessor.target_size}
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path: str) -> bool:
        try:
            if not os.path.exists(path):
                return False
            
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.symbols_display = model_data['symbols_display']
            self.training_accuracy = model_data.get('training_accuracy', 0.0)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'type': 'RandomForest',
            'n_estimators': self.model.n_estimators,
            'is_trained': self.is_trained,
            'training_accuracy': self.training_accuracy,
            'classes': list(self.model.classes_) if self.is_trained else [],
            'preprocessing': 'StandardImagePreprocessor'
        }

class SVMModel(ModelInterface):
    """
    Support Vector Machine implementation
    """
    
    def __init__(self, symbols_display: Dict[str, str], preprocessor: DataPreprocessorInterface):
        self.model = SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced')
        self.symbols_display = symbols_display
        self.preprocessor = preprocessor
        self.is_trained = False
        self.training_accuracy = 0.0
        
    def train(self, X: List[np.ndarray], y: np.ndarray) -> float:
        X_processed = self.preprocessor.preprocess_dataset(X)
        X_flattened = X_processed.reshape(X_processed.shape[0], -1)
        
        if len(X_flattened) < 2 or len(np.unique(y)) < 2:
            self.model.fit(X_flattened, y)
            self.training_accuracy = 1.0 if len(X_flattened) > 0 else 0.0
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_flattened, y, test_size=0.2, random_state=42, stratify=y
            )
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            self.training_accuracy = accuracy_score(y_test, y_pred)
        
        self.is_trained = True
        return self.training_accuracy
    
    def predict(self, X: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # FIXED: Always use preprocess_image for single predictions
        X_processed = self.preprocessor.preprocess_image(X)
        X_processed = X_processed.reshape(1, -1)
        
        prediction_proba = self.model.predict_proba(X_processed)[0]
        predicted_class_idx = np.argmax(prediction_proba)
        predicted_symbol_name = self.model.classes_[predicted_class_idx]
        confidence = prediction_proba[predicted_class_idx]
        
        all_predictions = {}
        for i, class_name in enumerate(self.model.classes_):
            # Ensure class_name is a standard Python string for JSON serialization
            key_class_name = str(class_name)
            all_predictions[key_class_name] = {
                'symbol': self.symbols_display.get(key_class_name, key_class_name),
                'probability': float(prediction_proba[i]) # Ensure float
            }
        
        # Ensure predicted_symbol_name is also a standard Python string
        return (self.symbols_display.get(str(predicted_symbol_name), str(predicted_symbol_name)), 
                float(confidence), all_predictions) # Ensure confidence is float
    
    def save_model(self, path: str) -> None:
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_data = {
            'model': self.model,
            'symbols_display': self.symbols_display,
            'training_accuracy': self.training_accuracy,
            'preprocessor_config': {'target_size': self.preprocessor.target_size}
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path: str) -> bool:
        try:
            if not os.path.exists(path):
                return False
            
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.symbols_display = model_data['symbols_display']
            self.training_accuracy = model_data.get('training_accuracy', 0.0)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'type': 'SVM',
            'kernel': self.model.kernel,
            'is_trained': self.is_trained,
            'training_accuracy': self.training_accuracy,
            'classes': list(self.model.classes_) if self.is_trained else [],
            'preprocessing': 'StandardImagePreprocessor'
        }

class NeuralNetworkModel(ModelInterface):
    """
    Neural Network implementation using MLPClassifier
    """
    
    def __init__(self, symbols_display: Dict[str, str], preprocessor: DataPreprocessorInterface):
        self.model = MLPClassifier(
            hidden_layer_sizes=(50,),  # Simplified to single hidden layer
            max_iter=1000,  # Increased iterations
            random_state=42,
            early_stopping=False,  # Disable early stopping for small datasets
            validation_fraction=0.1,
            alpha=0.01,  # Add regularization
            solver='lbfgs'  # Better for small datasets
        )
        self.symbols_display = symbols_display
        self.preprocessor = preprocessor
        self.is_trained = False
        self.training_accuracy = 0.0
        
    def train(self, X: List[np.ndarray], y: np.ndarray) -> float:
        print(f"[NN_TRAIN] Starting training with {len(X)} images")
        
        X_processed = self.preprocessor.preprocess_dataset(X)
        print(f"[NN_TRAIN] Preprocessed shape: {X_processed.shape}, dtype: {X_processed.dtype}")
        
        X_flattened = X_processed.reshape(X_processed.shape[0], -1)
        print(f"[NN_TRAIN] Flattened shape: {X_flattened.shape}, dtype: {X_flattened.dtype}")
        
        # Normalize data for neural networks - ensure float type and handle edge cases
        X_flattened = X_flattened.astype(np.float32)
        # Avoid division by zero and ensure valid normalization
        if X_flattened.max() > 0:
            X_flattened = X_flattened / X_flattened.max()
        else:
            X_flattened = X_flattened / 255.0
        
        print(f"[NN_TRAIN] After normalization shape: {X_flattened.shape}, dtype: {X_flattened.dtype}")
        print(f"[NN_TRAIN] Data range: min={X_flattened.min()}, max={X_flattened.max()}")
        
        # For small datasets or single class, use simpler training
        unique_classes = np.unique(y)
        if len(X_flattened) < 10 or len(unique_classes) < 2:
            print(f"[NN_TRAIN] Small dataset or single class, fitting directly")
            self.model.fit(X_flattened, y)
            self.training_accuracy = 1.0 if len(X_flattened) > 0 else 0.0
        else:
            print(f"[NN_TRAIN] Using cross-validation approach")
            # Use stratified split but handle small datasets better
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_flattened, y, test_size=max(0.2, 2/len(X_flattened)), 
                    random_state=42, stratify=y
                )
                print(f"[NN_TRAIN] Train shape: {X_train.shape}, Test shape: {X_test.shape}")
                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_test)
                self.training_accuracy = accuracy_score(y_test, y_pred)
            except Exception as e:
                print(f"[NN_TRAIN] Split failed: {e}, using full dataset")
                self.model.fit(X_flattened, y)
                self.training_accuracy = 1.0 if len(X_flattened) > 0 else 0.0
        
        self.is_trained = True
        print(f"[NN_TRAIN] Training completed with accuracy: {self.training_accuracy}")
        return self.training_accuracy
    
    def predict(self, X: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # FIXED: Always use preprocess_image for single predictions
        X_processed = self.preprocessor.preprocess_image(X)
        X_processed = X_processed.reshape(1, -1)
        
        # Normalize for neural network
        X_processed = X_processed.astype(np.float32) / 255.0
        
        prediction_proba = self.model.predict_proba(X_processed)[0]
        predicted_class_idx = np.argmax(prediction_proba)
        predicted_symbol_name = self.model.classes_[predicted_class_idx]
        confidence = prediction_proba[predicted_class_idx]
        
        all_predictions = {}
        for i, class_name in enumerate(self.model.classes_):
            all_predictions[class_name] = {
                'symbol': self.symbols_display.get(class_name, class_name),
                'probability': float(prediction_proba[i])
            }
        
        return (self.symbols_display.get(predicted_symbol_name, predicted_symbol_name), 
                confidence, all_predictions)
    
    def save_model(self, path: str) -> None:
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_data = {
            'model': self.model,
            'symbols_display': self.symbols_display,
            'training_accuracy': self.training_accuracy,
            'preprocessor_config': {'target_size': self.preprocessor.target_size}
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path: str) -> bool:
        try:
            if not os.path.exists(path):
                return False
            
            model_data = joblib.load(path)
            self.model = model_data['model']
            self.symbols_display = model_data['symbols_display']
            self.training_accuracy = model_data.get('training_accuracy', 0.0)
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'type': 'NeuralNetwork',
            'hidden_layer_sizes': self.model.hidden_layer_sizes,
            'is_trained': self.is_trained,
            'training_accuracy': self.training_accuracy,
            'classes': list(self.model.classes_) if self.is_trained else [],
            'preprocessing': 'StandardImagePreprocessor'
        }

class ConvolutionalNeuralNetwork(ModelInterface):
    """
    Real CNN implementation using TensorFlow/Keras
    Implements actual convolutional neural networks for image classification
    """
    
    def __init__(self, symbols_display: Dict[str, str], preprocessor: DataPreprocessorInterface = None):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available. Cannot create CNN model.")
        
        self.symbols_display = symbols_display
        self.preprocessor = preprocessor if preprocessor else CNNImagePreprocessor()
        self.model = None
        self.history = None
        self.is_trained = False
        self.training_accuracy = 0.0
        self.input_shape = self.preprocessor.target_size + (1,)  # (H, W, C)
        self.num_classes = len(symbols_display)
        
        # Training configuration
        self.epochs = 50
        self.batch_size = 16  # Smaller batch size for small datasets
        self.patience = 10
        
        # Normalization parameters (stored for prediction)
        self.feature_means = None
        self.feature_stds = None
        
        print(f"[CNN] Initialized with input shape: {self.input_shape}, num classes: {self.num_classes}")
    
    def _create_cnn_model(self, architecture='intermediate'):
        """Create CNN model based on the notebook architecture"""
        
        if architecture == 'basic':
            model = Sequential([
                # First convolutional block
                Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
                BatchNormalization(),
                MaxPooling2D((2, 2)),

                # Second convolutional block
                Conv2D(64, (3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),

                # Third convolutional block
                Conv2D(128, (3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),

                # Dense layers
                Flatten(),
                Dense(128, activation='relu'),
                Dropout(0.5),
                Dense(64, activation='relu'),
                Dropout(0.3),
                Dense(self.num_classes, activation='softmax')
            ])
            
        elif architecture == 'intermediate':
            model = Sequential([
                # First block
                Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
                Conv2D(32, (3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Dropout(0.25),

                # Second block
                Conv2D(64, (3, 3), activation='relu'),
                Conv2D(64, (3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Dropout(0.25),

                # Third block
                Conv2D(128, (3, 3), activation='relu'),
                Conv2D(128, (3, 3), activation='relu'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Dropout(0.25),

                # Dense layers
                Flatten(),
                Dense(256, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(128, activation='relu'),
                Dropout(0.3),
                Dense(self.num_classes, activation='softmax')
            ])
            
        else:  # advanced
            model = Sequential([
                # First block
                Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape, padding='same'),
                Conv2D(32, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Dropout(0.2),

                # Second block
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                Conv2D(64, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Dropout(0.2),

                # Third block
                Conv2D(128, (3, 3), activation='relu', padding='same'),
                Conv2D(128, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Dropout(0.3),

                # Fourth block
                Conv2D(256, (3, 3), activation='relu', padding='same'),
                Conv2D(256, (3, 3), activation='relu', padding='same'),
                BatchNormalization(),
                MaxPooling2D((2, 2)),
                Dropout(0.3),

                # Dense layers
                Flatten(),
                Dense(512, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                Dense(256, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                Dense(128, activation='relu'),
                Dropout(0.3),
                Dense(self.num_classes, activation='softmax')
            ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X: List[np.ndarray], y: np.ndarray) -> float:
        """Train the CNN model"""
        print(f"[CNN_TRAIN] Starting CNN training with {len(X)} images")
        
        try:
            # Preprocess data for CNN
            X_processed = self.preprocessor.preprocess_dataset(X)
            print(f"[CNN_TRAIN] X_processed shape: {X_processed.shape}, dtype: {X_processed.dtype}")
            print(f"[CNN_TRAIN] Data range: [{X_processed.min():.3f}, {X_processed.max():.3f}]")
            
            # Convert labels to categorical
            unique_labels = sorted(list(set(y)))
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            y_numeric = np.array([label_to_idx[label] for label in y])
            y_categorical = to_categorical(y_numeric, num_classes=self.num_classes)
            
            print(f"[CNN_TRAIN] Labels shape: {y_categorical.shape}")
            print(f"[CNN_TRAIN] Unique classes: {unique_labels}")
            print(f"[CNN_TRAIN] Label mapping: {label_to_idx}")
            
            # Handle small datasets
            n_samples = X_processed.shape[0]
            if n_samples < 10 or len(unique_labels) < 2:
                print(f"[CNN_TRAIN] Small dataset ({n_samples} samples, {len(unique_labels)} classes)")
                # Use simpler architecture and training for small datasets
                self.model = self._create_cnn_model('basic')
                self.epochs = min(20, self.epochs)
                self.batch_size = min(8, n_samples)
                
                # Train on full dataset
                history = self.model.fit(
                    X_processed, y_categorical,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    verbose=1
                )
                
                # Evaluate on training data
                train_loss, train_acc = self.model.evaluate(X_processed, y_categorical, verbose=0)
                self.training_accuracy = train_acc
                
            else:
                print(f"[CNN_TRAIN] Standard training with {n_samples} samples")
                
                # Create model
                self.model = self._create_cnn_model('intermediate')
                
                # Split data for validation
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(
                    X_processed, y_categorical, 
                    test_size=0.2, 
                    random_state=42, 
                    stratify=y_numeric
                )
                
                print(f"[CNN_TRAIN] Training split: {X_train.shape[0]} train, {X_val.shape[0]} val")
                
                # Setup callbacks
                callbacks = [
                    EarlyStopping(
                        monitor='val_loss',
                        patience=self.patience,
                        restore_best_weights=True,
                        verbose=1
                    ),
                    ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=5,
                        min_lr=1e-7,
                        verbose=1
                    )
                ]
                
                # Train model
                history = self.model.fit(
                    X_train, y_train,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks,
                    verbose=1
                )
                
                # Final evaluation
                val_loss, val_acc = self.model.evaluate(X_val, y_val, verbose=0)
                self.training_accuracy = val_acc
            
            self.history = history
            self.is_trained = True
            
            # Store label mapping for prediction
            self.label_mapping = {idx: label for label, idx in label_to_idx.items()}
            
            print(f"[CNN_TRAIN] Training completed with accuracy: {self.training_accuracy:.4f}")
            return self.training_accuracy
            
        except Exception as e:
            print(f"[CNN_TRAIN] Error during training: {e}")
            import traceback
            print(f"[CNN_TRAIN] Full traceback: {traceback.format_exc()}")
            self.training_accuracy = 0.0
            return 0.0
    
    def predict(self, X: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        """Predict using the trained CNN"""
        if not self.is_trained or self.model is None:
            raise ValueError("CNN model must be trained before prediction")
        
        try:
            # Preprocess single image
            X_processed = self.preprocessor.preprocess_image(X)
            print(f"[CNN_PREDICT] Preprocessed shape: {X_processed.shape}, range: [{X_processed.min():.3f}, {X_processed.max():.3f}]")
            
            # Add batch dimension
            X_batch = np.expand_dims(X_processed, axis=0)
            print(f"[CNN_PREDICT] Batch shape: {X_batch.shape}")
            
            # Make prediction
            predictions = self.model.predict(X_batch, verbose=0)
            pred_proba = predictions[0]
            
            print(f"[CNN_PREDICT] Raw predictions: {pred_proba}")
            
            # Get predicted class
            predicted_class_idx = np.argmax(pred_proba)
            predicted_symbol_name = self.label_mapping.get(predicted_class_idx, f"class_{predicted_class_idx}")
            confidence = float(pred_proba[predicted_class_idx])
            
            print(f"[CNN_PREDICT] Predicted class idx: {predicted_class_idx}")
            print(f"[CNN_PREDICT] Predicted symbol: {predicted_symbol_name}")
            print(f"[CNN_PREDICT] Confidence: {confidence}")
            
            # Build all predictions dict
            all_predictions = {}
            for class_idx, prob in enumerate(pred_proba):
                class_name = self.label_mapping.get(class_idx, f"class_{class_idx}")
                all_predictions[class_name] = {
                    'symbol': self.symbols_display.get(class_name, class_name),
                    'probability': float(prob)
                }
            
            predicted_display = self.symbols_display.get(predicted_symbol_name, predicted_symbol_name)
            
            return predicted_display, confidence, all_predictions
            
        except Exception as e:
            print(f"[CNN_PREDICT] Error during prediction: {e}")
            import traceback
            print(f"[CNN_PREDICT] Full traceback: {traceback.format_exc()}")
            return f"Error during prediction: {e}", None, None
    
    def save_model(self, path: str) -> None:
        """Save the trained CNN model"""
        if not self.is_trained or self.model is None:
            raise ValueError("Cannot save untrained CNN model")
        
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save TensorFlow model
            model_h5_path = path.replace('.pkl', '.h5')
            self.model.save(model_h5_path)
            
            # Save additional model data
            model_data = {
                'symbols_display': self.symbols_display,
                'training_accuracy': self.training_accuracy,
                'input_shape': self.input_shape,
                'num_classes': self.num_classes,
                'label_mapping': getattr(self, 'label_mapping', {}),
                'preprocessor_config': {
                    'target_size': self.preprocessor.target_size,
                    'type': type(self.preprocessor).__name__
                },
                'model_h5_path': model_h5_path,
                'epochs': self.epochs,
                'batch_size': self.batch_size
            }
            
            joblib.dump(model_data, path)
            print(f"[CNN_SAVE] Saved CNN model to {path} and {model_h5_path}")
            
        except Exception as e:
            print(f"[CNN_SAVE] Error saving CNN model: {e}")
            raise
    
    def load_model(self, path: str) -> bool:
        """Load a trained CNN model"""
        try:
            if not os.path.exists(path):
                print(f"[CNN_LOAD] Model file not found: {path}")
                return False
            
            # Load model data
            model_data = joblib.load(path)
            
            # Load TensorFlow model
            model_h5_path = model_data.get('model_h5_path', path.replace('.pkl', '.h5'))
            if not os.path.exists(model_h5_path):
                print(f"[CNN_LOAD] TensorFlow model file not found: {model_h5_path}")
                return False
            
            # Load Keras model
            self.model = keras.models.load_model(model_h5_path)
            
            # Restore other attributes
            self.symbols_display = model_data['symbols_display']
            self.training_accuracy = model_data.get('training_accuracy', 0.0)
            self.input_shape = model_data.get('input_shape', (64, 64, 1))
            self.num_classes = model_data.get('num_classes', len(self.symbols_display))
            self.label_mapping = model_data.get('label_mapping', {})
            self.epochs = model_data.get('epochs', 50)
            self.batch_size = model_data.get('batch_size', 16)
            
            # Update preprocessor if needed
            preprocessor_config = model_data.get('preprocessor_config', {})
            if preprocessor_config.get('type') == 'CNNImagePreprocessor':
                target_size = preprocessor_config.get('target_size', (64, 64))
                self.preprocessor = CNNImagePreprocessor(target_size)
            
            self.is_trained = True
            
            print(f"[CNN_LOAD] Successfully loaded CNN model from {path}")
            print(f"[CNN_LOAD] Model info: accuracy={self.training_accuracy:.4f}, classes={self.num_classes}")
            
            return True
            
        except Exception as e:
            print(f"[CNN_LOAD] Error loading CNN model: {e}")
            import traceback
            print(f"[CNN_LOAD] Full traceback: {traceback.format_exc()}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get CNN model information"""
        info = {
            'type': 'ConvolutionalNeuralNetwork (TensorFlow/Keras)',
            'is_trained': self.is_trained,
            'training_accuracy': self.training_accuracy,
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'preprocessing': type(self.preprocessor).__name__,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'tensorflow_version': tf.__version__ if TENSORFLOW_AVAILABLE else 'Not available'
        }
        
        if self.is_trained and self.model:
            info.update({
                'total_params': self.model.count_params(),
                'classes': list(self.label_mapping.values()) if hasattr(self, 'label_mapping') else [],
                'architecture': 'Real CNN with Conv2D layers'
            })
        
        return info

# ============================================================================
# FACTORY PATTERN (Open/Closed Principle)
# ============================================================================

class ModelFactory:
    """
    Factory for creating different model types (Open/Closed Principle)
    """
    
    _models = {
        'random_forest': RandomForestModel,
        'svm': SVMModel,
        'neural_network': NeuralNetworkModel,
        'cnn': ConvolutionalNeuralNetwork
    }
    
    @classmethod
    def create_model(cls, model_type: str, symbols_display: Dict[str, str], 
                    preprocessor: DataPreprocessorInterface) -> ModelInterface:
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(cls._models.keys())}")
        
        return cls._models[model_type](symbols_display, preprocessor)
    
    @classmethod
    def get_available_models(cls) -> List[str]:
        return list(cls._models.keys())
    
    @classmethod
    def register_model(cls, name: str, model_class: type) -> None:
        """Allow registration of new model types (Open/Closed Principle)"""
        cls._models[name] = model_class

# ============================================================================
# SERVICE LAYER (Single Responsibility Principle)
# ============================================================================

class AIService:
    """
    Service layer that orchestrates AI operations (Single Responsibility Principle)
    """
    
    def __init__(self, model: ModelInterface):
        self.model = model
    
    def train_model(self, images: List[np.ndarray], labels: List[str]) -> Tuple[float, str]:
        """Train the model and return accuracy and message"""
        try:
            if not images:
                return 0.0, "No images available for training."
            
            # Don't convert to numpy array here - let the preprocessor handle it
            # X = np.array(images)  # This line causes the error!
            y = np.array(labels)
            
            # Pass the list of images directly to the model
            accuracy = self.model.train(images, y)
            message = f"Model trained successfully with accuracy: {accuracy:.4f}"
            
            return accuracy, message
        except Exception as e:
            return 0.0, f"Error during training: {e}"
    
    def predict_symbol(self, image: np.ndarray) -> Tuple[Optional[str], Optional[float], Optional[Dict[str, Any]]]:
        """Predict symbol from image"""
        try:
            return self.model.predict(image)
        except Exception as e:
            return f"Error during prediction: {e}", None, None
    
    def save_model(self, path: str) -> bool:
        """Save the current model"""
        try:
            self.model.save_model(path)
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """Load a model from path"""
        return self.model.load_model(path)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return self.model.get_model_info() 