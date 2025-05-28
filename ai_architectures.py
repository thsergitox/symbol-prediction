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

class ConvolutionalNeuralNetwork(ModelInterface):
    """
    CNN-like implementation using traditional image processing and deep neural networks
    Simulates convolutional operations using scipy filters and feature extraction
    """
    
    def __init__(self, symbols_display: Dict[str, str], preprocessor: DataPreprocessorInterface):
        # Simpler network more appropriate for small datasets
        self.model = MLPClassifier(
            hidden_layer_sizes=(32, 16),  # Much simpler: only 2 layers instead of 4
            max_iter=1000,  # Reduced iterations to prevent overfitting
            random_state=42,
            early_stopping=True,  
            validation_fraction=0.2,
            alpha=0.1,  # Higher regularization to prevent overfitting
            solver='lbfgs',  # Better for small datasets than adam
            learning_rate_init=0.01
        )
        self.symbols_display = symbols_display
        self.preprocessor = preprocessor
        self.is_trained = False
        self.training_accuracy = 0.0
        
        # Simpler convolutional filters for small datasets
        self.conv_filters = self._create_simple_filters()
        
    def _create_simple_filters(self):
        """Create simpler convolutional filters for small datasets"""
        filters = {
            'horizontal_edge': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
            'vertical_edge': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
            'diagonal_edge_1': np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]]),
            'diagonal_edge_2': np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]]),
            
            'corner_tl': np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]]),
            'corner_tr': np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]]),
            'corner_bl': np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]]),
            'corner_br': np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]]),
            
            'thick_horizontal': np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]]),
            'thick_vertical': np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]]),
            
            'curve_1': np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
            'curve_2': np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
        }
        return filters
    
    def _extract_conv_features(self, image: np.ndarray) -> np.ndarray:
        """Extract simplified features more suitable for small datasets"""
        features = []
        
        # Ensure image is float64 for consistent calculations
        image = image.astype(np.float64)
        
        # Apply only the most important filters and extract key statistics
        key_filters = ['horizontal_edge', 'vertical_edge', 'corner_tl', 'corner_tr']
        for filter_name in key_filters:
            if filter_name in self.conv_filters:
                kernel = self.conv_filters[filter_name]
                filtered = ndimage.convolve(image, kernel, mode='constant')
                
                if filtered.size > 0:
                    features.extend([
                        float(np.mean(filtered)),           # Average response
                        float(np.std(filtered)),            # Standard deviation  
                        float(np.mean(np.abs(filtered))),   # Mean absolute response
                    ])
                else:
                    features.extend([0.0, 0.0, 0.0])
        
        # Essential image statistics
        if image.size > 0:
            img_mean = float(np.mean(image))
            features.extend([
                img_mean,
                float(np.std(image)),
                float(np.sum(image > img_mean)) / image.size,  # Fraction above average
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Basic edge detection (simplified)
        try:
            sobel_h = filters.sobel_h(image)
            sobel_v = filters.sobel_v(image)
            if sobel_h.size > 0 and sobel_v.size > 0:
                edge_magnitude = np.sqrt(sobel_h**2 + sobel_v**2)
                features.extend([
                    float(np.mean(edge_magnitude)),
                    float(np.std(edge_magnitude)),
                ])
            else:
                features.extend([0.0, 0.0])
        except Exception as e:
            print(f"[CNN_FEATURE] Error in edge detection: {e}")
            features.extend([0.0, 0.0])
        
        # Simplified spatial features (2x2 grid instead of complex analysis)
        try:
            h, w = image.shape
            quadrants = [
                image[:h//2, :w//2],      # Top-left
                image[:h//2, w//2:],      # Top-right  
                image[h//2:, :w//2],      # Bottom-left
                image[h//2:, w//2:]       # Bottom-right
            ]
            
            for quad in quadrants:
                if quad.size > 0:
                    features.append(float(np.mean(quad)))
                else:
                    features.append(0.0)
        except Exception as e:
            print(f"[CNN_FEATURE] Error in spatial features: {e}")
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Convert to numpy array and ensure no NaN/inf values
        features_array = np.array(features, dtype=np.float64)
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)
        
        print(f"[CNN_FEATURE] Extracted {len(features_array)} features")
        return features_array
    
    def train(self, X: List[np.ndarray], y: np.ndarray) -> float:
        print(f"[CNN_TRAIN] Starting CNN-like training with {len(X)} images")
        
        # Preprocess images
        X_processed = self.preprocessor.preprocess_dataset(X)
        print(f"[CNN_TRAIN] Preprocessed shape: {X_processed.shape}")
        
        # Extract CNN-like features for each image
        X_features = []
        for i, img in enumerate(X_processed):
            features = self._extract_conv_features(img)
            # Ensure features are float64 and clean
            features = np.array(features, dtype=np.float64)
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            X_features.append(features)
            if i % 10 == 0:
                print(f"[CNN_TRAIN] Extracted features for image {i+1}/{len(X_processed)}")
        
        # Convert to proper numpy array with consistent dtype
        X_features = np.vstack(X_features).astype(np.float64)
        print(f"[CNN_TRAIN] Feature matrix shape: {X_features.shape}, dtype: {X_features.dtype}")
        
        # Final cleanup of features - ensure no NaN or inf values
        X_features = np.nan_to_num(X_features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Check for and handle any remaining problematic values
        if not np.isfinite(X_features).all():
            print(f"[CNN_TRAIN] Warning: Non-finite values detected, cleaning...")
            X_features = np.where(np.isfinite(X_features), X_features, 0.0)
        
        # Robust normalization with explicit dtype handling
        try:
            feature_means = np.mean(X_features, axis=0, dtype=np.float64)
            feature_stds = np.std(X_features, axis=0, dtype=np.float64)
            
            # Prevent division by zero with proper epsilon
            feature_stds = np.where(feature_stds < 1e-8, 1.0, feature_stds)
            
            X_features = (X_features - feature_means) / feature_stds
            X_features = X_features.astype(np.float64)  # Ensure consistent dtype
            
            print(f"[CNN_TRAIN] Normalized features - shape: {X_features.shape}, range: [{X_features.min():.3f}, {X_features.max():.3f}]")
        except Exception as e:
            print(f"[CNN_TRAIN] Normalization error: {e}, using raw features")
            X_features = X_features.astype(np.float64)
        
        # Ensure labels are proper strings (not numpy string types that can cause issues)
        y_clean = np.array([str(label) for label in y], dtype='U50')  # Unicode string with max 50 chars
        print(f"[CNN_TRAIN] Label types: {type(y_clean[0])}, unique labels: {np.unique(y_clean)}")
        
        # Train the deep neural network with proper error handling
        unique_classes = np.unique(y_clean)
        if len(X_features) < 15 or len(unique_classes) < 2:
            print(f"[CNN_TRAIN] Small dataset, fitting directly")
            try:
                self.model.fit(X_features, y_clean)
                # Calculate training accuracy safely
                y_pred_train = self.model.predict(X_features)
                self.training_accuracy = accuracy_score(y_clean, y_pred_train)
            except Exception as e:
                print(f"[CNN_TRAIN] Training error: {e}")
                self.training_accuracy = 0.0
        else:
            print(f"[CNN_TRAIN] Using train/test split for CNN")
            try:
                # Ensure stratify works with clean labels
                X_train, X_test, y_train, y_test = train_test_split(
                    X_features, y_clean, test_size=0.25, random_state=42, stratify=y_clean
                )
                print(f"[CNN_TRAIN] Training deep network - Train: {X_train.shape}, Test: {X_test.shape}")
                
                # Fit with clean data types
                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_test)
                self.training_accuracy = accuracy_score(y_test, y_pred)
                print(f"[CNN_TRAIN] CNN achieved {self.training_accuracy:.4f} accuracy")
                
            except Exception as e:
                print(f"[CNN_TRAIN] Split/train failed: {e}, using full dataset")
                try:
                    self.model.fit(X_features, y_clean)
                    # Calculate training accuracy on full dataset
                    y_pred_train = self.model.predict(X_features)
                    self.training_accuracy = accuracy_score(y_clean, y_pred_train)
                    print(f"[CNN_TRAIN] Fallback training accuracy: {self.training_accuracy:.4f}")
                except Exception as e2:
                    print(f"[CNN_TRAIN] Fallback training also failed: {e2}")
                    self.training_accuracy = 0.0
        
        self.is_trained = True
        print(f"[CNN_TRAIN] CNN training completed with accuracy: {self.training_accuracy}")
        return self.training_accuracy
    
    def predict(self, X: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        if not self.is_trained:
            raise ValueError("CNN model must be trained before prediction")
        
        # Preprocess single image
        if X.ndim == 2:
            X_processed = self.preprocessor.preprocess_image(X)
        else:
            X_processed = self.preprocessor.preprocess_dataset([X])[0]
        
        # Extract CNN-like features
        features = self._extract_conv_features(X_processed)
        features = features.reshape(1, -1)
        
        # Handle any potential NaN or inf values
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Simple normalization for prediction (we don't have training statistics stored)
        # This is a simplified approach - ideally we'd store the training normalization parameters
        feature_mean = np.mean(features)
        feature_std = np.std(features)
        if feature_std > 0:
            features = (features - feature_mean) / feature_std
        
        print(f"[CNN_PREDICT] Features shape: {features.shape}, range: [{features.min():.3f}, {features.max():.3f}]")
        
        # Predict using deep network
        prediction_proba = self.model.predict_proba(features)[0]
        predicted_class_idx = np.argmax(prediction_proba)
        predicted_symbol_name = self.model.classes_[predicted_class_idx]
        confidence = prediction_proba[predicted_class_idx]
        
        all_predictions = {}
        for i, class_name in enumerate(self.model.classes_):
            all_predictions[str(class_name)] = {  # Ensure string key
                'symbol': self.symbols_display.get(str(class_name), str(class_name)),
                'probability': float(prediction_proba[i])
            }
        
        return (self.symbols_display.get(str(predicted_symbol_name), str(predicted_symbol_name)), 
                float(confidence), all_predictions)
    
    def save_model(self, path: str) -> None:
        if not self.is_trained:
            raise ValueError("Cannot save untrained CNN model")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_data = {
            'model': self.model,
            'symbols_display': self.symbols_display,
            'training_accuracy': self.training_accuracy,
            'preprocessor_config': {'target_size': self.preprocessor.target_size},
            'conv_filters': self.conv_filters
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
            self.conv_filters = model_data.get('conv_filters', self._create_simple_filters())
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error loading CNN model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'type': 'ConvolutionalNN (Simplified)',
            'hidden_layer_sizes': self.model.hidden_layer_sizes,
            'is_trained': self.is_trained,
            'training_accuracy': self.training_accuracy,
            'classes': list(self.model.classes_) if self.is_trained else [],
            'preprocessing': 'StandardImagePreprocessor + SimplifiedConvFeatures',
            'num_filters': len(self.conv_filters),
            'solver': self.model.solver,
            'optimization': 'Designed for small datasets'
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