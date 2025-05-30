"""
Simplified TensorFlow/Keras CNN Model wrapper that doesn't require TensorFlow
This is a mock implementation for testing without TensorFlow installed
"""

import os
import numpy as np
import joblib
from typing import Dict, List, Tuple, Any, Optional
from ai_architectures import ModelInterface, DataPreprocessorInterface, ConvolutionalNeuralNetwork, StandardImagePreprocessor

class CNNPreprocessor(DataPreprocessorInterface):
    """
    Preprocessor specifically for the CNN model (64x64 images)
    """
    
    def __init__(self, target_size: Tuple[int, int] = (64, 64)):
        self.target_size = target_size
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        # Use StandardImagePreprocessor but with 64x64 target
        preprocessor = StandardImagePreprocessor(target_size=self.target_size)
        return preprocessor.preprocess_image(image)
    
    def preprocess_dataset(self, images: List[np.ndarray]) -> np.ndarray:
        preprocessor = StandardImagePreprocessor(target_size=self.target_size)
        return preprocessor.preprocess_dataset(images)


class SimplifiedTensorFlowCNN(ModelInterface):
    """
    Simplified CNN that mimics TensorFlow CNN interface without requiring TensorFlow
    """
    
    def __init__(self, symbols_display: Dict[str, str], preprocessor: DataPreprocessorInterface = None):
        # Use the existing simplified CNN underneath
        self.cnn_model = ConvolutionalNeuralNetwork(symbols_display, preprocessor or CNNPreprocessor())
        self.symbols_display = symbols_display
        self.preprocessor = preprocessor or CNNPreprocessor(target_size=(64, 64))
        self.is_trained = False
        self.training_accuracy = 0.95  # Mock high accuracy for pre-trained model
        self.classes_ = ['alpha', 'beta', 'epsilon']
        self.model_path_h5 = './model/mejor_modelo.h5'
        
    def train(self, X: List[np.ndarray], y: np.ndarray) -> float:
        """Mock training - we assume model is pre-trained"""
        print("[MOCK_TF_CNN] This is a pre-trained model simulation. Training skipped.")
        self.is_trained = True
        return self.training_accuracy
    
    def predict(self, X: np.ndarray) -> Tuple[str, float, Dict[str, Any]]:
        # For demonstration, use the underlying simplified CNN
        # In reality, this would load and use the .h5 model
        if not self.is_trained:
            # Auto-load for pre-trained model
            self.load_model(self.model_path_h5)
        
        # Use the underlying CNN model for prediction
        return self.cnn_model.predict(X)
    
    def save_model(self, path: str) -> None:
        """Save model metadata"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'model_path_h5': self.model_path_h5,
            'symbols_display': self.symbols_display,
            'training_accuracy': self.training_accuracy,
            'classes': self.classes_,
            'preprocessor_config': {'target_size': self.preprocessor.target_size},
            'model_type': 'tensorflow_cnn',
            'is_mock': True  # Indicate this is a mock implementation
        }
        joblib.dump(model_data, path)
        print(f"[MOCK_TF_CNN] Model metadata saved to {path}")
    
    def load_model(self, path: str) -> bool:
        """Load model (mock implementation)"""
        try:
            # First try to load the underlying CNN model
            cnn_pkl_path = os.path.join(os.path.dirname(path), 'modelo_cnn.pkl')
            if os.path.exists(cnn_pkl_path):
                self.cnn_model.load_model(cnn_pkl_path)
                print(f"[MOCK_TF_CNN] Loaded underlying CNN model from {cnn_pkl_path}")
            
            # Load metadata if it's a .pkl file
            if path.endswith('.pkl') and os.path.exists(path):
                model_data = joblib.load(path)
                self.symbols_display = model_data.get('symbols_display', self.symbols_display)
                self.training_accuracy = model_data.get('training_accuracy', 0.95)
                self.classes_ = model_data.get('classes', self.classes_)
                self.model_path_h5 = model_data.get('model_path_h5', self.model_path_h5)
            
            self.is_trained = True
            print(f"[MOCK_TF_CNN] Model loaded (mock mode)")
            return True
            
        except Exception as e:
            print(f"[MOCK_TF_CNN] Error loading model: {e}")
            # Even if loading fails, mark as trained for demo
            self.is_trained = True
            return True
    
    def get_model_info(self) -> Dict[str, Any]:
        return {
            'type': 'TensorFlow CNN (Mock)',
            'architecture': 'Convolutional Neural Network',
            'is_trained': self.is_trained,
            'training_accuracy': self.training_accuracy,
            'classes': self.classes_,
            'preprocessing': 'CNNPreprocessor (64x64)',
            'framework': 'TensorFlow/Keras (Simulated)',
            'model_summary': '15 layers, ~300k parameters (simulated)',
            'total_params': 300000,
            'input_shape': '(64, 64, 1)',
            'note': 'This is a mock implementation without actual TensorFlow'
        }


# Use the simplified version as the wrapper
TensorFlowCNNWrapper = SimplifiedTensorFlowCNN 