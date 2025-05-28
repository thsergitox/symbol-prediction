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

# ============================================================================
# ABSTRACTIONS (Dependency Inversion Principle)
# ============================================================================

class ModelInterface(ABC):
    """
    Interface for all ML models (Dependency Inversion Principle)
    """
    
    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> float:
        """Train the model and return accuracy"""
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
        from skimage import transform
        
        # Handle different image formats
        if image.ndim == 3 and image.shape[2] == 4:  # RGBA
            image = image[:, :, 3]  # Use alpha channel
        elif image.ndim == 3 and image.shape[2] == 3:  # RGB
            image = np.mean(image, axis=2).astype(np.uint8)
        elif image.ndim == 2:  # Already grayscale
            pass
        
        # Resize image
        image_resized = transform.resize(
            image, self.target_size,
            anti_aliasing=True, preserve_range=True
        ).astype(np.uint8)
        
        return image_resized
    
    def preprocess_dataset(self, images: List[np.ndarray]) -> np.ndarray:
        processed_images = []
        for img in images:
            processed_img = self.preprocess_image(img)
            processed_images.append(processed_img)
        
        return np.array(processed_images)

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
        
    def train(self, X: np.ndarray, y: np.ndarray) -> float:
        # Preprocess data
        X_processed = self.preprocessor.preprocess_dataset(X)
        X_flattened = X_processed.reshape(X_processed.shape[0], -1)
        
        # Handle small datasets
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
        
    def train(self, X: np.ndarray, y: np.ndarray) -> float:
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
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        self.symbols_display = symbols_display
        self.preprocessor = preprocessor
        self.is_trained = False
        self.training_accuracy = 0.0
        
    def train(self, X: np.ndarray, y: np.ndarray) -> float:
        X_processed = self.preprocessor.preprocess_dataset(X)
        X_flattened = X_processed.reshape(X_processed.shape[0], -1)
        
        # Normalize data for neural networks
        X_flattened = X_flattened / 255.0
        
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
        
        # Normalize for neural network
        X_processed = X_processed / 255.0
        
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
            
            X = np.array(images)
            y = np.array(labels)
            
            accuracy = self.model.train(X, y)
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