import os
import tempfile
import base64
import glob
import numpy as np
from flask import Flask, request, redirect, send_file, render_template, jsonify, session
from skimage import io, transform
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = 'os.urandom(24)' # Added a secret key for session management

# Configuration
STATIC_FOLDER = os.path.join(os.getcwd(), 'static')
TEMPLATES_FOLDER = os.path.join(os.getcwd(), 'templates')
MODEL_FILENAME = 'modelo_simbolos.pkl'
LAST_TRAINING_FILE = 'last_training_time.txt'
SYMBOLS = ['alpha', 'beta', 'epsilon'] # α, β, ε
SYMBOLS_DISPLAY = {'alpha': 'α', 'beta': 'β', 'epsilon': 'ε'}
IMAGE_DIM = (100, 100) # Standard image dimension for the model
MIN_TRAINING_INTERVAL = 5  # Minimum minutes between trainings

# Ensure data directories exist
for symbol in SYMBOLS:
    if not os.path.exists(symbol):
        os.makedirs(symbol)
        print(f"[INIT] Created directory for symbol: {symbol}")

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
    with open(LAST_TRAINING_FILE, 'w') as f:
        f.write(content)

def train_and_save_model_adapted(model_filename=MODEL_FILENAME):
    print("[TRAIN] Starting model training...")
    try:
        images = []
        labels = []
        for symbol_name in SYMBOLS:
            print(f"[TRAIN] Processing images for symbol {SYMBOLS_DISPLAY[symbol_name]} ({symbol_name})...")
            filelist = glob.glob(f'{symbol_name}/*.png')
            print(f"[TRAIN] Found {len(filelist)} images in {symbol_name}/")
            if not filelist:
                print(f"[WARN] No images found in folder {symbol_name}")
                continue
            for img_path in filelist:
                try:
                    img = io.imread(img_path)
                    if img.ndim == 3 and img.shape[2] == 4: # RGBA
                        img = img[:, :, 3] # Use alpha channel
                    elif img.ndim == 3 and img.shape[2] == 3: # RGB
                        img = np.mean(img, axis=2).astype(np.uint8)
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
        joblib.dump(model, model_filename)
        print(f"[TRAIN] Model saved to: {model_filename}")
        update_last_training_time(current_accuracy)
        print("[TRAIN] Training process finished.")
        return current_accuracy, f"Model trained successfully with accuracy: {current_accuracy:.4f}"
    except Exception as e:
        print(f"[ERROR] Error during model training: {e}")
        return None, f"Error during training: {e}"

def predict_letter_adapted(image_data_base64, model_filename=MODEL_FILENAME):
    print("[PREDICT] Starting prediction...")
    try:
        if not os.path.exists(model_filename):
            print("[PREDICT] Model not found. Please train the model first.")
            return "Model not found. Please train the model first.", None, None
        model = joblib.load(model_filename)
        print(f"[PREDICT] Model loaded from {model_filename}")
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
                'symbol': SYMBOLS_DISPLAY[class_name],
                'probability': float(prediction_proba[i])
            }
        
        print(f"[PREDICT] Predicted symbol: {SYMBOLS_DISPLAY[predicted_symbol_name]} (class: {predicted_symbol_name}) with confidence {confidence:.2f}")
        return SYMBOLS_DISPLAY[predicted_symbol_name], confidence, all_predictions
    except Exception as e:
        print(f"[ERROR] Error during prediction: {e}")
        return f"Error during prediction: {e}", None, None

@app.route("/")
def index():
    print("[ROUTE] GET /")
    return render_template('index.html')

@app.route("/create-dataset", methods=['GET'])
def create_dataset_page():
    print("[ROUTE] GET /create-dataset")
    return render_template('create_dataset.html', symbols=SYMBOLS, symbols_display=SYMBOLS_DISPLAY)

@app.route("/upload-image", methods=['POST'])
def upload_image():
    print("[ROUTE] POST /upload-image")
    try:
        img_data = request.form.get('myImage').replace("data:image/png;base64,", "")
        symbol_to_draw = request.form.get('symbol')
        print(f"[UPLOAD] Received image for symbol: {symbol_to_draw}")
        if not symbol_to_draw or symbol_to_draw not in SYMBOLS:
            print(f"[ERROR] Invalid symbol specified: {symbol_to_draw}")
            return "Invalid symbol specified.", 400
        
        symbol_folder = os.path.join(os.getcwd(), symbol_to_draw)
        if not os.path.exists(symbol_folder):
             os.makedirs(symbol_folder)
             print(f"[UPLOAD] Created directory: {symbol_folder}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{symbol_to_draw}_{timestamp}.png"
        filepath = os.path.join(symbol_folder, filename)
        
        with open(filepath, "wb") as fh:
            fh.write(base64.b64decode(img_data))
        print(f"[UPLOAD] Image saved to {filepath}")
        return redirect("/create-dataset")
    except Exception as e:
        print(f"[ERROR] Error uploading image: {e}")
        return "Error uploading image.", 500

@app.route("/train", methods=['GET', 'POST'])
def train_page():
    print(f"[ROUTE] {'POST' if request.method == 'POST' else 'GET'} /train")
    message = ""
    
    # Get last training time and accuracy
    last_trained_time, last_accuracy = get_last_training_info()
    
    # Determine if training is allowed based on time
    can_train_now, minutes_to_wait = True, None
    if last_trained_time:
        time_since_last_training = datetime.now() - last_trained_time
        minutes_since_last_training = time_since_last_training.total_seconds() / 60
        if minutes_since_last_training < MIN_TRAINING_INTERVAL:
            can_train_now = False
            minutes_to_wait = MIN_TRAINING_INTERVAL - minutes_since_last_training
    
    if request.method == 'POST':
        print("[TRAIN] Training requested by user.")
        if can_train_now:
            training_accuracy, msg = train_and_save_model_adapted()
            message = msg
            print(f"[TRAIN] Training result: {msg}")
            
            # After training, update info for display
            last_trained_time, last_accuracy = get_last_training_info()
            can_train_now, minutes_to_wait = True, None # Reset, as training just happened
            # Recalculate wait time for immediate next check (though button might be disabled by can_train_now=True implies min_wait is 0)
            time_since_last_training = datetime.now() - last_trained_time
            minutes_since_last_training = time_since_last_training.total_seconds() / 60
            if minutes_since_last_training < MIN_TRAINING_INTERVAL:
                can_train_now = False # Should be false right after training if interval is strict
                minutes_to_wait = MIN_TRAINING_INTERVAL - minutes_since_last_training
            else:
                can_train_now = True
                minutes_to_wait = None
        else:
            message = f"Entrenamiento no permitido. Espere {minutes_to_wait:.1f} minutos más."
            print(f"[TRAIN] Training rejected: {message}")
    
    dataset_stats = {}
    total_images = 0
    for symbol_name in SYMBOLS:
        count = len(glob.glob(f'{symbol_name}/*.png'))
        dataset_stats[SYMBOLS_DISPLAY[symbol_name]] = count
        total_images += count
    print(f"[TRAIN] Dataset stats: {dataset_stats}, Total images: {total_images}")
    
    model_age_warning = False
    if last_trained_time and (datetime.now() - last_trained_time > timedelta(minutes=30)): # General age warning
        model_age_warning = True
        print("[TRAIN] Model is older than 30 minutes.")
        
    return render_template('train.html', 
                        stats=dataset_stats, 
                        total_images=total_images, 
                        message=message, 
                        last_trained_time=last_trained_time, 
                        last_accuracy=last_accuracy,
                        model_age_warning=model_age_warning,
                        can_train_now=can_train_now,
                        minutes_to_wait=minutes_to_wait if minutes_to_wait is not None else MIN_TRAINING_INTERVAL,
                        min_training_interval=MIN_TRAINING_INTERVAL)

@app.route("/predict", methods=['GET', 'POST'])
def predict_page():
    print(f"[ROUTE] {'POST' if request.method == 'POST' else 'GET'} /predict")
    
    prediction_result = None
    confidence_score = None
    all_predictions = None
    
    if request.method == 'POST':
        image_data = request.form.get('myImage')
        print(f"[PREDICT] Received image for prediction. Data present: {bool(image_data)}")
        if image_data:
            prediction_result, confidence_score, all_predictions = predict_letter_adapted(image_data)
            print(f"[PREDICT] Prediction result: {prediction_result}, Confidence: {confidence_score}, All: {all_predictions}")
            if not isinstance(prediction_result, str) or "Error" not in prediction_result:
                session['last_prediction_data'] = {
                    'prediction': prediction_result,
                    'confidence': confidence_score,
                    'all_predictions': all_predictions,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'image_data_url': image_data # Store the submitted canvas image
                }
                print("[PREDICT][SESSION] Stored current prediction and image in session.")


    last_trained_time, last_accuracy = get_last_training_info()
    model_age_warning = False
    model_exists = os.path.exists(MODEL_FILENAME)
    
    last_trained_display = "No entrenado. Por favor, entrena el modelo primero." # Default message

    if model_exists:
        if last_trained_time:
            last_trained_display = last_trained_time # Will be formatted in template if datetime
            if (datetime.now() - last_trained_time > timedelta(minutes=30)):
                model_age_warning = True
                print("[PREDICT] Model is older than 30 minutes.")
        else: # Model file exists, but no training time record
            last_trained_display = "Desconocido (modelo existe, pero no hay registro de entrenamiento)"
            model_age_warning = True # Treat as potentially old if no training date
            print("[PREDICT] Model exists but no training time record found.")
    else:
        print("[PREDICT] No model found.")
        # last_trained_display remains the default "No entrenado..."

    # Determine what to display: current POST result or session's last prediction for GET
    display_prediction = prediction_result
    display_confidence = confidence_score
    display_all_predictions = all_predictions
    

    return render_template('predict.html', 
                         prediction=display_prediction, 
                         confidence=display_confidence, 
                         all_predictions=display_all_predictions,
                         last_trained_time_obj=last_trained_time, 
                         last_trained_display=last_trained_display, 
                         last_accuracy=last_accuracy, 
                         model_age_warning=model_age_warning, 
                         model_exists=model_exists,
                         )

if __name__ == "__main__":
    print("[INIT] Starting Flask app on 0.0.0.0:5000...")
    app.run(debug=True, host='0.0.0.0')
