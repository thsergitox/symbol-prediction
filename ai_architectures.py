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
from skimage import transform

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
        
        # Handle different image formats and convert to grayscale
        if image.ndim == 3 and image.shape[2] == 4:  # RGBA
            image = image[:, :, 3]  # Use alpha channel for drawings
        elif image.ndim == 3 and image.shape[2] == 3:  # RGB
            image = np.mean(image, axis=2).astype(np.uint8)
        elif image.ndim == 2:  # Already grayscale
            image = image.astype(np.uint8)
        else:
            raise ValueError(f"Unsupported image format: shape {image.shape}")
        
        # Ensure we have a 2D grayscale image
        if image.ndim != 2:
            raise ValueError(f"Could not convert to 2D grayscale: shape {image.shape}")
        
        # Resize image to target size
        try:
            image_resized = transform.resize(
                image, self.target_size,
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
        
        # Preprocess single image
        if X.ndim == 2:  # Single image
            X_processed = self.preprocessor.preprocess_image(X)
            X_processed = X_processed.reshape(1, -1)
        else:
            X_processed = self.preprocessor.preprocess_dataset([X])
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
        
        if X.ndim == 2:
            X_processed = self.preprocessor.preprocess_image(X)
            X_processed = X_processed.reshape(1, -1)
        else:
            X_processed = self.preprocessor.preprocess_dataset([X])
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
        
        if X.ndim == 2:
            X_processed = self.preprocessor.preprocess_image(X)
            X_processed = X_processed.reshape(1, -1)
        else:
            X_processed = self.preprocessor.preprocess_dataset([X])
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
        'neural_network': NeuralNetworkModel
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