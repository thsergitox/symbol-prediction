import os
import glob
import numpy as np
from skimage import io, transform
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from datetime import datetime
import tempfile
import base64

# Import configurations from config.py
from config import (
    MODEL_FILENAME, LAST_TRAINING_FILE, SYMBOLS, SYMBOLS_DISPLAY, 
    IMAGE_DIM, MIN_TRAINING_INTERVAL, DATASET_BASE_PATH
)

def get_last_training_info():
    print("[LOG] Checking last training info...")
    if os.path.exists(LAST_TRAINING_FILE):
        with open(LAST_TRAINING_FILE, 'r') as f:
            line = f.read().strip()
            if line:
                parts = line.split('|')
                if len(parts) == 2:
                    timestamp_str, accuracy_str = parts
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str)
                        accuracy = float(accuracy_str)
                        print(f"[LOG] Last training: {timestamp_str}, Accuracy: {accuracy:.4f}")
                        return timestamp, accuracy
                    except ValueError:
                        print(f"[ERROR] Could not parse line from {LAST_TRAINING_FILE}: {line}")
                else: # Old format, just timestamp
                    try:
                        timestamp = datetime.fromisoformat(line)
                        print(f"[LOG] Last training time found (old format): {line}")
                        return timestamp, None # No accuracy info
                    except ValueError:
                        print(f"[ERROR] Could not parse old format timestamp from {LAST_TRAINING_FILE}: {line}")                        
    print("[LOG] No last training info found.")
    return None, None

def can_train_again():
    last_trained_time, _ = get_last_training_info()
    if last_trained_time is None:
        return True, None
    
    now = datetime.now()
    time_since_last_training = now - last_trained_time
    minutes_since_last_training = time_since_last_training.total_seconds() / 60
    
    if minutes_since_last_training >= MIN_TRAINING_INTERVAL:
        return True, None
    else:
        minutes_to_wait = MIN_TRAINING_INTERVAL - minutes_since_last_training
        return False, minutes_to_wait

def update_last_training_time(accuracy=None):
    now_iso = datetime.now().isoformat()
    if accuracy is not None:
        content = f"{now_iso}|{accuracy:.4f}"
        print(f"[LOG] Updating last training time and accuracy: {now_iso}, Acc: {accuracy:.4f}")
    else:
        content = now_iso # Fallback or if accuracy is not available
        print(f"[LOG] Updating last training time (no accuracy): {now_iso}")
    os.makedirs(os.path.dirname(LAST_TRAINING_FILE), exist_ok=True) # Ensure directory exists
    with open(LAST_TRAINING_FILE, 'w') as f:
        f.write(content)

def train_and_save_model_adapted(model_filename_param=MODEL_FILENAME):
    print("[TRAIN] Starting model training...")
    try:
        images = []
        labels = []
        for symbol_name in SYMBOLS:
            print(f"[TRAIN] Processing images for symbol {SYMBOLS_DISPLAY.get(symbol_name, symbol_name)} ({symbol_name})...")
            symbol_folder = os.path.join(DATASET_BASE_PATH, symbol_name)
            filelist = glob.glob(os.path.join(symbol_folder, '*.png'))
            print(f"[TRAIN] Found {len(filelist)} images in {symbol_folder}/")
            if not filelist:
                print(f"[WARN] No images found in folder {symbol_folder}")
                continue
            for img_path in filelist:
                try:
                    img = io.imread(img_path)
                    if img.ndim == 3 and img.shape[2] == 4: # RGBA
                        img = img[:, :, 3] # Use alpha channel
                    elif img.ndim == 3 and img.shape[2] == 3: # RGB
                        img = np.mean(img, axis=2).astype(np.uint8)
                    # Add handling for already grayscale images if necessary
                    elif img.ndim == 2:
                        pass # Already grayscale
                    else:
                        print(f"[WARN] Skipping image with unhandled dimensions: {img_path} shape {img.shape}")
                        continue
                    
                    img_resized = transform.resize(img, IMAGE_DIM,
                                                 anti_aliasing=True, preserve_range=True).astype(np.uint8)
                    images.append(img_resized)
                    labels.append(symbol_name)
                except Exception as e:
                    print(f"[ERROR] Error loading or processing image {img_path}: {e}")
        if not images:
            print("[ERROR] No images found to train the model.")
            return None, "No images available for training."
        X = np.array(images)
        y = np.array(labels)
        n_samples = X.shape[0]
        X_reshaped = X.reshape(n_samples, -1)
        print(f"[TRAIN] Dataset: {n_samples} images, {len(np.unique(y))} classes: {np.unique(y)}")
        
        current_accuracy = 0.0 # Default accuracy

        if n_samples < 2 or len(np.unique(y)) < 2:
            print("[WARN] Not enough samples or classes to train a model and split data.")
            if n_samples < 1:
                return None, "Not enough samples to train."
            model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            model.fit(X_reshaped, y)
            current_accuracy = 1.0 if n_samples > 0 else 0.0
            print(f"[TRAIN] Model trained on all data (no split). Accuracy set to {current_accuracy:.4f}.")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_reshaped, y, test_size=0.2, random_state=42, stratify=y
            )
            print(f"[TRAIN] Training set: {X_train.shape}, Test set: {X_test.shape}")
            model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            model.fit(X_train, y_train)
            current_accuracy = model.score(X_test, y_test)
            print(f"[TRAIN] Model accuracy on test set: {current_accuracy:.4f}")
        
        os.makedirs(os.path.dirname(model_filename_param), exist_ok=True) # Ensure model directory exists
        joblib.dump(model, model_filename_param)
        print(f"[TRAIN] Model saved to: {model_filename_param}")
        update_last_training_time(current_accuracy)
        print("[TRAIN] Training process finished.")
        return current_accuracy, f"Model trained successfully with accuracy: {current_accuracy:.4f}"
    except Exception as e:
        print(f"[ERROR] Error during model training: {e}")
        return None, f"Error during training: {e}"

def predict_letter_adapted(image_data_base64, model_filename_param=MODEL_FILENAME):
    print("[PREDICT] Starting prediction...")
    try:
        if not os.path.exists(model_filename_param):
            print("[PREDICT] Model not found. Please train the model first.")
            return "Model not found. Please train the model first.", None, None
        model = joblib.load(model_filename_param)
        print(f"[PREDICT] Model loaded from {model_filename_param}")
        img_data = image_data_base64.replace("data:image/png;base64,", "")
        with tempfile.NamedTemporaryFile(delete=True, suffix='.png') as tmp:
            tmp.write(base64.b64decode(img_data))
            tmp.flush()
            image = io.imread(tmp.name)
        print(f"[PREDICT] Image loaded for prediction. Shape: {image.shape}")
        if image.ndim == 3 and image.shape[2] == 4:
            image = image[:, :, 3]
        elif image.ndim == 3 and image.shape[2] == 3:
             image = np.mean(image, axis=2).astype(np.uint8)
        # Add handling for already grayscale images if necessary
        elif image.ndim == 2:
            pass # Already grayscale
        else:
            print(f"[ERROR] Prediction image has unhandled dimensions: {image.shape}")
            return "Error processing image for prediction.", None, None

        image_resized = transform.resize(image, IMAGE_DIM,
                                     anti_aliasing=True, preserve_range=True).astype(np.uint8)
        image_vector = image_resized.reshape(1, -1)
        print(f"[PREDICT] Image preprocessed for model. Shape: {image_vector.shape}")
        prediction_proba = model.predict_proba(image_vector)[0]
        predicted_class_idx = np.argmax(prediction_proba)
        predicted_symbol_name = model.classes_[predicted_class_idx]
        confidence = prediction_proba[predicted_class_idx]
        
        all_predictions = {}
        for i, class_name in enumerate(model.classes_):
            all_predictions[class_name] = {
                'symbol': SYMBOLS_DISPLAY.get(class_name, class_name),
                'probability': float(prediction_proba[i])
            }
        
        print(f"[PREDICT] Predicted symbol: {SYMBOLS_DISPLAY.get(predicted_symbol_name, predicted_symbol_name)} (class: {predicted_symbol_name}) with confidence {confidence:.2f}")
        return SYMBOLS_DISPLAY.get(predicted_symbol_name, predicted_symbol_name), confidence, all_predictions
    except Exception as e:
        print(f"[ERROR] Error during prediction: {e}")
        return f"Error during prediction: {e}", None, None 