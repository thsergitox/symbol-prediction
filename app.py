import os
import glob
from flask import Flask, request, render_template, session, redirect
from datetime import datetime

# Import configurations from config.py
from config import (
    SYMBOLS, SYMBOLS_DISPLAY, MODEL_FILENAME, 
    MIN_TRAINING_INTERVAL, STATIC_FOLDER, TEMPLATES_FOLDER,
    APP_DIR, DATASET_BASE_PATH, MODEL_BASE_PATH, DB_FOLDER
)

# Import AI utility functions
from ai import (
    get_last_training_info, train_and_save_model_adapted, 
    predict_letter_adapted, can_train_again
)

# Import router from save.py
from save import register_routes

app = Flask(__name__, static_folder=STATIC_FOLDER, template_folder=TEMPLATES_FOLDER)
app.secret_key = os.urandom(24) # For session management

# Register save routes using our custom router
register_routes(app)

# --- Directory Initialization ---
print(f"[INIT] Ensuring directories exist using base path: {APP_DIR}")
# Ensure data directories exist using paths from config
if not os.path.exists(DATASET_BASE_PATH):
    os.makedirs(DATASET_BASE_PATH)
    print(f"[INIT] Created dataset base directory: {DATASET_BASE_PATH}")

for symbol in SYMBOLS:
    symbol_dir = os.path.join(DATASET_BASE_PATH, symbol)
    if not os.path.exists(symbol_dir):
        os.makedirs(symbol_dir)
        print(f"[INIT] Created directory for symbol: {symbol} at {symbol_dir}")

if not os.path.exists(MODEL_BASE_PATH):
    os.makedirs(MODEL_BASE_PATH)
    print(f"[INIT] Created model directory: {MODEL_BASE_PATH}")

if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)
    print(f"[INIT] Created db directory: {DB_FOLDER}")

# --- Main Application Routes --- 

@app.route("/")
def index():
    print("[ROUTE] GET /")
    return render_template('index.html')

@app.route("/create-dataset", methods=['GET'])
def create_dataset_page():
    print("[ROUTE] GET /create-dataset")
    return render_template('create_dataset.html', symbols=SYMBOLS, symbols_display=SYMBOLS_DISPLAY)

@app.route("/train", methods=['GET', 'POST'])
def train_page():
    print(f"[ROUTE] {'POST' if request.method == 'POST' else 'GET'} /train")
    message = ""
    
    last_trained_time, last_accuracy = get_last_training_info()
    # Use can_train_again() from ai.py
    can_train_now, minutes_to_wait = can_train_again()
        
    if request.method == 'POST':
        print("[TRAIN] Training requested by user.")
        if can_train_now:
            # Use train_and_save_model_adapted from ai.py
            training_accuracy, msg = train_and_save_model_adapted()
            message = msg
            print(f"[TRAIN] Training result: {msg}")
            
            last_trained_time, last_accuracy = get_last_training_info() # Update info after training
            can_train_now, minutes_to_wait = can_train_again() # Recheck ability to train
        else:
            message = f"Entrenamiento no permitido. Espere {minutes_to_wait:.1f} minutos más."
            print(f"[TRAIN] Training rejected: {message}")
    
    dataset_stats = {}
    total_images = 0
    for symbol_name in SYMBOLS:
        symbol_folder = os.path.join(DATASET_BASE_PATH, symbol_name) # Use config path
        count = len(glob.glob(os.path.join(symbol_folder, '*.png')))
        dataset_stats[SYMBOLS_DISPLAY.get(symbol_name, symbol_name)] = count
        total_images += count
    print(f"[TRAIN] Dataset stats: {dataset_stats}, Total images: {total_images}")
    
    return render_template('train.html', 
                        stats=dataset_stats, 
                        total_images=total_images, 
                        message=message, 
                        last_trained_time=last_trained_time, 
                        last_accuracy=last_accuracy,
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
            # Use predict_letter_adapted from ai.py
            prediction_result, confidence_score, all_predictions = predict_letter_adapted(image_data)
            print(f"[PREDICT] Prediction result: {prediction_result}, Confidence: {confidence_score}, All: {all_predictions}")
            if not isinstance(prediction_result, str) or "Error" not in prediction_result:
                session['last_prediction_data'] = {
                    'prediction': prediction_result,
                    'confidence': confidence_score,
                    'all_predictions': all_predictions, 
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'image_data_url': image_data 
                }
                print("[PREDICT][SESSION] Stored current prediction and image in session.")

    last_trained_time, last_accuracy = get_last_training_info()
    model_exists = os.path.exists(MODEL_FILENAME) # Use config path
    
    last_trained_display = "No entrenado. Por favor, entrena el modelo primero." 

    if model_exists:
        if last_trained_time:
            last_trained_display = last_trained_time 
        else: 
            last_trained_display = "Desconocido (modelo existe, pero no hay registro de entrenamiento)"
            print("[PREDICT] Model exists but no training time record found.")
    else:
        print("[PREDICT] No model found.")

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
                         model_exists=model_exists)

if __name__ == "__main__":
    print(f"[INIT] Starting Flask app on 0.0.0.0:5000...")
    app.run(debug=True, host='0.0.0.0')
