import os
import glob
import json
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from skimage import io
import numpy as np

# Import the new architecture
from ai_architectures import (
    ModelInterface, DataPreprocessorInterface, StandardImagePreprocessor, CNNImagePreprocessor,
    ModelFactory, AIService
)

# Import configurations
from config import (
    SYMBOLS, SYMBOLS_DISPLAY, DATASET_BASE_PATH, MODEL_BASE_PATH,
    MIN_TRAINING_INTERVAL, DB_FOLDER
)

class ModelManager:
    """
    Manager for handling multiple AI models (Single Responsibility Principle)
    Coordinates model selection, training, and persistence
    """
    
    def __init__(self):
        self.current_model_type = 'random_forest'  # Default model
        self.preprocessor = StandardImagePreprocessor()
        self.ai_service: Optional[AIService] = None
        self.models_info_file = os.path.join(DB_FOLDER, 'models_info.json')
        self.current_model_file = os.path.join(DB_FOLDER, 'current_model.txt')
        
        # Initialize directories
        self._ensure_directories()
        
        # Load current model selection
        self._load_current_model_selection()
        
        # Initialize AI service with current model
        self._initialize_current_model()
    
    def _ensure_directories(self):
        """Ensure all necessary directories exist"""
        os.makedirs(MODEL_BASE_PATH, exist_ok=True)
        os.makedirs(DB_FOLDER, exist_ok=True)
    
    def _load_current_model_selection(self):
        """Load the currently selected model type"""
        if os.path.exists(self.current_model_file):
            try:
                with open(self.current_model_file, 'r') as f:
                    model_type = f.read().strip()
                    if model_type in ModelFactory.get_available_models():
                        self.current_model_type = model_type
                        print(f"[MODEL_MANAGER] Loaded current model selection: {model_type}")
                    else:
                        print(f"[MODEL_MANAGER] Invalid model type in file: {model_type}, using default")
            except Exception as e:
                print(f"[MODEL_MANAGER] Error loading current model selection: {e}")
    
    def _save_current_model_selection(self):
        """Save the currently selected model type"""
        try:
            with open(self.current_model_file, 'w') as f:
                f.write(self.current_model_type)
            print(f"[MODEL_MANAGER] Saved current model selection: {self.current_model_type}")
        except Exception as e:
            print(f"[MODEL_MANAGER] Error saving current model selection: {e}")
    
    def _initialize_current_model(self):
        """Initialize the AI service with the current model"""
        try:
            # Choose appropriate preprocessor based on model type
            if self.current_model_type == 'cnn':
                preprocessor = CNNImagePreprocessor()
                print(f"[MODEL_MANAGER] Using CNNImagePreprocessor for CNN model")
            else:
                preprocessor = self.preprocessor  # Use the standard preprocessor
                print(f"[MODEL_MANAGER] Using StandardImagePreprocessor for {self.current_model_type} model")
            
            model = ModelFactory.create_model(
                self.current_model_type,
                SYMBOLS_DISPLAY,
                preprocessor
            )
            self.ai_service = AIService(model)
            
            # Try to load existing model
            model_path = self.get_model_path(self.current_model_type)
            if os.path.exists(model_path):
                success = self.ai_service.load_model(model_path)
                if success:
                    print(f"[MODEL_MANAGER] Loaded existing {self.current_model_type} model")
                else:
                    print(f"[MODEL_MANAGER] Failed to load existing {self.current_model_type} model")
            else:
                print(f"[MODEL_MANAGER] No existing {self.current_model_type} model found")
                
        except Exception as e:
            print(f"[MODEL_MANAGER] Error initializing current model: {e}")
            import traceback
            print(f"[MODEL_MANAGER] Full traceback: {traceback.format_exc()}")
    
    def get_model_path(self, model_type: str) -> str:
        """Get the file path for a specific model type"""
        return os.path.join(MODEL_BASE_PATH, f'modelo_{model_type}.pkl')
    
    def get_available_models(self) -> List[str]:
        """Get list of available model types"""
        return ModelFactory.get_available_models()
    
    def get_current_model_type(self) -> str:
        """Get the currently selected model type"""
        return self.current_model_type
    
    def switch_model(self, new_model_type: str) -> bool:
        """
        Switch to a different model type
        Returns True if successful, False otherwise
        """
        if new_model_type not in ModelFactory.get_available_models():
            print(f"[MODEL_MANAGER] Invalid model type: {new_model_type}")
            return False
        
        if new_model_type == self.current_model_type:
            print(f"[MODEL_MANAGER] Already using {new_model_type}")
            return True
        
        try:
            # Save current model selection
            old_model_type = self.current_model_type
            self.current_model_type = new_model_type
            self._save_current_model_selection()
            
            # Initialize new model
            self._initialize_current_model()
            
            print(f"[MODEL_MANAGER] Successfully switched from {old_model_type} to {new_model_type}")
            return True
            
        except Exception as e:
            print(f"[MODEL_MANAGER] Error switching to {new_model_type}: {e}")
            # Revert on error
            self.current_model_type = old_model_type if 'old_model_type' in locals() else 'random_forest'
            return False
    
    def get_all_models_info(self) -> Dict[str, Any]:
        """Get information about all available models"""
        models_info = {}
        
        for model_type in ModelFactory.get_available_models():
            model_path = self.get_model_path(model_type)
            model_exists = os.path.exists(model_path)
            
            model_info = {
                'type': model_type,
                'exists': model_exists,
                'is_current': model_type == self.current_model_type,
                'path': model_path
            }
            
            # If model exists, try to get more detailed info
            if model_exists:
                try:
                    # Use appropriate preprocessor for each model type
                    if model_type == 'cnn':
                        temp_preprocessor = CNNImagePreprocessor()
                    else:
                        temp_preprocessor = self.preprocessor
                    
                    temp_model = ModelFactory.create_model(model_type, SYMBOLS_DISPLAY, temp_preprocessor)
                    if temp_model.load_model(model_path):
                        detailed_info = temp_model.get_model_info()
                        model_info.update(detailed_info)
                except Exception as e:
                    print(f"[MODEL_MANAGER] Error getting info for {model_type}: {e}")
                    model_info['error'] = str(e)
            
            models_info[model_type] = model_info
        
        return models_info
    
    def get_current_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        if self.ai_service is None:
            return {'error': 'No model initialized'}
        
        try:
            return self.ai_service.get_model_info()
        except Exception as e:
            return {'error': str(e)}
    
    def load_dataset_images(self) -> Tuple[List[np.ndarray], List[str]]:
        """Load all dataset images and labels"""
        images = []
        labels = []
        
        for symbol_name in SYMBOLS:
            print(f"[MODEL_MANAGER] Loading images for symbol {SYMBOLS_DISPLAY.get(symbol_name, symbol_name)} ({symbol_name})...")
            symbol_folder = os.path.join(DATASET_BASE_PATH, symbol_name)
            filelist = glob.glob(os.path.join(symbol_folder, '*.png'))
            print(f"[MODEL_MANAGER] Found {len(filelist)} images in {symbol_folder}/")
            
            if not filelist:
                print(f"[WARN] No images found in folder {symbol_folder}")
                continue
            
            for img_path in filelist:
                try:
                    img = io.imread(img_path)
                    
                    # Ensure consistent image format
                    if img.ndim == 3 and img.shape[2] == 4:  # RGBA
                        # Use alpha channel for binary images (drawings)
                        img = img[:, :, 3]
                    elif img.ndim == 3 and img.shape[2] == 3:  # RGB
                        # Convert to grayscale
                        img = np.mean(img, axis=2).astype(np.uint8)
                    elif img.ndim == 2:  # Already grayscale
                        img = img.astype(np.uint8)
                    else:
                        print(f"[WARN] Unexpected image format for {img_path}: shape {img.shape}")
                        continue
                    
                    # Ensure 2D grayscale image
                    if img.ndim != 2:
                        print(f"[WARN] Could not convert {img_path} to 2D grayscale, skipping")
                        continue
                    
                    images.append(img)
                    labels.append(symbol_name)
                    print(f"[MODEL_MANAGER] Loaded {img_path} with shape {img.shape}")
                    
                except Exception as e:
                    print(f"[ERROR] Error loading image {img_path}: {e}")
        
        print(f"[MODEL_MANAGER] Total loaded: {len(images)} images")
        return images, labels
    
    def train_current_model(self) -> Tuple[float, str]:
        """Train the current model"""
        if self.ai_service is None:
            return 0.0, "No model initialized"
        
        print(f"[MODEL_MANAGER] Attempting to train {self.current_model_type} model...")
        
        # Load dataset
        images, labels = self.load_dataset_images()
        
        if not images:
            print("[MODEL_MANAGER_TRAIN] No images loaded from dataset. Cannot train.")
            return 0.0, "No images available for training."
        
        print(f"[MODEL_MANAGER_TRAIN] Loaded {len(images)} images for training.")
        if labels:
            unique_labels, counts = np.unique(labels, return_counts=True)
            print(f"[MODEL_MANAGER_TRAIN] Label distribution: {dict(zip(unique_labels, counts))}")
        else:
            print("[MODEL_MANAGER_TRAIN] No labels loaded. Cannot train.")
            return 0.0, "No labels available for training."

        # Train model
        print(f"[MODEL_MANAGER_TRAIN] Passing {len(images)} images and {len(labels)} labels to AIService for {self.current_model_type}.")
        accuracy, message = self.ai_service.train_model(images, labels)
        print(f"[MODEL_MANAGER_TRAIN] Received from AIService: accuracy={accuracy}, message='{message}'")
        
        # Define model path
        model_path = self.get_model_path(self.current_model_type)

        # Attempt to delete existing model file before saving a new one
        if os.path.exists(model_path):
            try:
                os.remove(model_path)
                print(f"[MODEL_MANAGER_TRAIN] Deleted existing model file: {model_path}")
            except OSError as e:
                print(f"[MODEL_MANAGER_TRAIN] Error deleting existing model file {model_path}: {e}")
                # Decide if you want to proceed if deletion fails, or return an error
                # For now, we'll just log and continue

        # Save model if training was successful (and accuracy is not a placeholder for error)
        if accuracy > 0.0 and "Error" not in message: # Ensure it's a real accuracy
            if self.ai_service.save_model(model_path):
                print(f"[MODEL_MANAGER_TRAIN] Successfully saved {self.current_model_type} model to {model_path} with accuracy {accuracy}")
                self._update_model_training_info(accuracy)
            else:
                print(f"[MODEL_MANAGER_TRAIN] Failed to save {self.current_model_type} model after training.")
        else:
            print(f"[MODEL_MANAGER_TRAIN] Model training for {self.current_model_type} did not result in positive accuracy or an error occurred. Model not saved. Reported accuracy: {accuracy}")
        
        return accuracy, message
    
    def predict_with_current_model(self, image_data_base64: str) -> Tuple[Optional[str], Optional[float], Optional[Dict[str, Any]]]:
        """Predict using the current model"""
        if self.ai_service is None:
            return "No model initialized", None, None
        
        try:
            # Convert base64 to image - FIXED: Handle different image formats
            import base64
            import tempfile
            import re
            
            print(f"[MODEL_MANAGER] Raw image data length: {len(image_data_base64)}")
            print(f"[MODEL_MANAGER] Image data prefix: {image_data_base64[:50]}...")
            
            # Remove data URL prefix more robustly
            if image_data_base64.startswith('data:'):
                # Use regex to remove any data URL prefix (png, jpeg, webp, etc.)
                img_data = re.sub(r'^data:image/[^;]+;base64,', '', image_data_base64)
            else:
                img_data = image_data_base64
            
            print(f"[MODEL_MANAGER] Cleaned base64 data length: {len(img_data)}")
            
            # Decode base64 and save to temporary file
            try:
                decoded_data = base64.b64decode(img_data)
                print(f"[MODEL_MANAGER] Decoded image data length: {len(decoded_data)}")
            except Exception as e:
                print(f"[MODEL_MANAGER] Error decoding base64: {e}")
                return "Error decoding image data", None, None
            
            with tempfile.NamedTemporaryFile(delete=True, suffix='.png') as tmp:
                tmp.write(decoded_data)
                tmp.flush()
                
                # Load image using skimage
                image = io.imread(tmp.name)
            
            print(f"[MODEL_MANAGER] === PREDICTION DEBUG ===")
            print(f"[MODEL_MANAGER] Current model type: {self.current_model_type}")
            print(f"[MODEL_MANAGER] Model info: {self.ai_service.get_model_info()}")
            print(f"[MODEL_MANAGER] Image shape: {image.shape}")
            print(f"[MODEL_MANAGER] Image dtype: {image.dtype}")
            print(f"[MODEL_MANAGER] Image data range: [{image.min()}, {image.max()}]")
            
            # Check if image is valid
            if image.size == 0:
                return "Empty image received", None, None
            
            prediction, confidence, all_predictions = self.ai_service.predict_symbol(image)
            
            print(f"[MODEL_MANAGER] Prediction result: {prediction}")
            print(f"[MODEL_MANAGER] Confidence: {confidence}")
            print(f"[MODEL_MANAGER] All predictions keys: {list(all_predictions.keys()) if all_predictions else None}")
            if all_predictions:
                for class_name, pred_data in all_predictions.items():
                    print(f"[MODEL_MANAGER] {class_name}: {pred_data}")
            print(f"[MODEL_MANAGER] === END PREDICTION DEBUG ===")
            
            return prediction, confidence, all_predictions
            
        except Exception as e:
            import traceback
            print(f"[MODEL_MANAGER] Error in prediction: {e}")
            print(f"[MODEL_MANAGER] Full traceback: {traceback.format_exc()}")
            return f"Error during prediction: {e}", None, None
    
    def _update_model_training_info(self, accuracy: float):
        """Update training information for the current model"""
        try:
            # Load existing info
            models_info = {}
            if os.path.exists(self.models_info_file):
                with open(self.models_info_file, 'r') as f:
                    models_info = json.load(f)
            
            # Update current model info
            models_info[self.current_model_type] = {
                'last_trained': datetime.now().isoformat(),
                'accuracy': accuracy,
                'training_count': models_info.get(self.current_model_type, {}).get('training_count', 0) + 1
            }
            
            # Save updated info
            with open(self.models_info_file, 'w') as f:
                json.dump(models_info, f, indent=2)
            
            print(f"[MODEL_MANAGER] Updated training info for {self.current_model_type}")
            
        except Exception as e:
            print(f"[MODEL_MANAGER] Error updating training info: {e}")
    
    def get_last_training_info(self) -> Tuple[Optional[datetime], Optional[float]]:
        """Get last training information for current model"""
        try:
            if not os.path.exists(self.models_info_file):
                return None, None
            
            with open(self.models_info_file, 'r') as f:
                models_info = json.load(f)
            
            current_info = models_info.get(self.current_model_type, {})
            if 'last_trained' in current_info:
                timestamp = datetime.fromisoformat(current_info['last_trained'])
                accuracy = current_info.get('accuracy')
                return timestamp, accuracy
            
            return None, None
            
        except Exception as e:
            print(f"[MODEL_MANAGER] Error getting training info: {e}")
            return None, None
    
    def can_train_again(self) -> Tuple[bool, Optional[float]]:
        """Check if training is allowed - no time restrictions"""
        # Removed time-based training restrictions
        return True, None
    
    def model_exists(self, model_type: Optional[str] = None) -> bool:
        """Check if a model file exists"""
        if model_type is None:
            model_type = self.current_model_type
        
        model_path = self.get_model_path(model_type)
        return os.path.exists(model_path)

# Global model manager instance
model_manager = ModelManager() 