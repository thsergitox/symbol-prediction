import os
import glob
from flask import Flask, request, render_template, session, redirect, jsonify
from datetime import datetime

# Import configurations from config.py
from config import (
    SYMBOLS, SYMBOLS_DISPLAY, MIN_TRAINING_INTERVAL, STATIC_FOLDER, 
    TEMPLATES_FOLDER, APP_DIR, DATASET_BASE_PATH, MODEL_BASE_PATH, DB_FOLDER
)

# Import the new AI architecture
from ai_manager import model_manager

# Import router from save.py
from save import register_routes

app = Flask(__name__, static_folder=STATIC_FOLDER, template_folder=TEMPLATES_FOLDER)
app.secret_key = os.urandom(24)  # For session management

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

@app.route("/view-dataset", methods=['GET'])
def view_dataset_page():
    print("[ROUTE] GET /view-dataset")
    
    # Get all images from dataset
    dataset_images = {}
    total_images = 0
    
    for symbol_name in SYMBOLS:
        symbol_folder = os.path.join(DATASET_BASE_PATH, symbol_name)
        if os.path.exists(symbol_folder):
            image_files = glob.glob(os.path.join(symbol_folder, '*.png'))
            dataset_images[symbol_name] = []
            
            for img_path in image_files:
                # Get relative path for web serving
                rel_path = os.path.relpath(img_path, APP_DIR)
                dataset_images[symbol_name].append({
                    'path': rel_path,
                    'filename': os.path.basename(img_path),
                    'full_path': img_path
                })
                total_images += 1
        else:
            dataset_images[symbol_name] = []
    
    return render_template('view_dataset.html', 
                         dataset_images=dataset_images, 
                         symbols_display=SYMBOLS_DISPLAY,
                         total_images=total_images)

@app.route("/delete-image", methods=['POST'])
def delete_image():
    print("[ROUTE] POST /delete-image")
    
    try:
        symbol = request.form.get('symbol')
        filename = request.form.get('filename')
        
        if not symbol or not filename:
            return jsonify({'success': False, 'error': 'Symbol and filename required'}), 400
        
        if symbol not in SYMBOLS:
            return jsonify({'success': False, 'error': 'Invalid symbol'}), 400
        
        # Construct file path
        img_path = os.path.join(DATASET_BASE_PATH, symbol, filename)
        
        # Security check - ensure the file is within our dataset directory
        if not os.path.abspath(img_path).startswith(os.path.abspath(DATASET_BASE_PATH)):
            return jsonify({'success': False, 'error': 'Invalid file path'}), 400
        
        # Delete the file
        if os.path.exists(img_path):
            os.remove(img_path)
            print(f"[DELETE] Deleted image: {img_path}")
            return jsonify({'success': True, 'message': f'Image {filename} deleted successfully'})
        else:
            return jsonify({'success': False, 'error': 'File not found'}), 404
            
    except Exception as e:
        print(f"[DELETE] Error deleting image: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route("/dataset/<symbol>/<filename>")
def serve_dataset_image(symbol, filename):
    """Serve images from the dataset folders"""
    print(f"[ROUTE] GET /dataset/{symbol}/{filename}")
    
    # Security check
    if symbol not in SYMBOLS:
        return "Invalid symbol", 404
    
    # Construct safe file path
    symbol_folder = os.path.join(DATASET_BASE_PATH, symbol)
    file_path = os.path.join(symbol_folder, filename)
    
    # Security check - ensure the file is within our dataset directory
    if not os.path.abspath(file_path).startswith(os.path.abspath(DATASET_BASE_PATH)):
        return "Invalid file path", 403
    
    # Check if file exists
    if not os.path.exists(file_path):
        return "File not found", 404
    
    # Serve the file
    try:
        from flask import send_file
        return send_file(file_path, mimetype='image/png')
    except Exception as e:
        print(f"[ERROR] Error serving image {file_path}: {e}")
        return "Error serving file", 500

@app.route("/train", methods=['GET', 'POST'])
def train_page():
    print(f"[ROUTE] {'POST' if request.method == 'POST' else 'GET'} /train")
    message = ""
    
    # Get training info from model manager
    last_trained_time, last_accuracy = model_manager.get_last_training_info()
    can_train_now, minutes_to_wait = model_manager.can_train_again()
    
    # Get current model info
    current_model_type = model_manager.get_current_model_type()
    available_models = model_manager.get_available_models()
    all_models_info = model_manager.get_all_models_info()
        
    if request.method == 'POST':
        action = request.form.get('action', 'train')
        
        if action == 'switch_model':
            new_model_type = request.form.get('model_type')
            if new_model_type and new_model_type != current_model_type:
                success = model_manager.switch_model(new_model_type)
                if success:
                    message = f"Cambiado exitosamente a modelo: {new_model_type}"
                    current_model_type = new_model_type
                    # Update training info for the new model
                    last_trained_time, last_accuracy = model_manager.get_last_training_info()
                    can_train_now, minutes_to_wait = model_manager.can_train_again()
                    all_models_info = model_manager.get_all_models_info()
                else:
                    message = f"Error al cambiar a modelo: {new_model_type}"
            else:
                message = "Tipo de modelo inválido o igual al actual"
                
        elif action == 'train':
            print("[TRAIN] Training requested by user.")
            if can_train_now:
                training_accuracy, msg = model_manager.train_current_model()
                message = msg
                print(f"[TRAIN] Training result: {msg}")
                
                # Update info after training
                last_trained_time, last_accuracy = model_manager.get_last_training_info()
                can_train_now, minutes_to_wait = model_manager.can_train_again()
                all_models_info = model_manager.get_all_models_info()
            else:
                message = f"Entrenamiento no permitido. Espere {minutes_to_wait:.1f} minutos más."
                print(f"[TRAIN] Training rejected: {message}")
    
    # Calculate dataset statistics
    dataset_stats = {}
    total_images = 0
    for symbol_name in SYMBOLS:
        symbol_folder = os.path.join(DATASET_BASE_PATH, symbol_name)
        count = len(glob.glob(os.path.join(symbol_folder, '*.png')))
        dataset_stats[SYMBOLS_DISPLAY.get(symbol_name, symbol_name)] = count
        total_images += count
    print(f"[TRAIN] Dataset stats: {dataset_stats}, Total images: {total_images}")
    
    return render_template('train_new.html', 
                        stats=dataset_stats, 
                        total_images=total_images, 
                        message=message, 
                        last_trained_time=last_trained_time, 
                        last_accuracy=last_accuracy,
                        can_train_now=can_train_now,
                        minutes_to_wait=minutes_to_wait if minutes_to_wait is not None else MIN_TRAINING_INTERVAL,
                        min_training_interval=MIN_TRAINING_INTERVAL,
                        current_model_type=current_model_type,
                        available_models=available_models,
                        all_models_info=all_models_info)

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
            # Use new model manager for prediction
            prediction_result, confidence_score, all_predictions = model_manager.predict_with_current_model(image_data)
            print(f"[PREDICT] Prediction result: {prediction_result}, Confidence: {confidence_score}, All: {all_predictions}")
            
            if not isinstance(prediction_result, str) or "Error" not in prediction_result:
                session['last_prediction_data'] = {
                    'prediction': prediction_result,
                    'confidence': confidence_score,
                    'all_predictions': all_predictions, 
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'image_data_url': image_data,
                    'model_type': model_manager.get_current_model_type()
                }
                print("[PREDICT][SESSION] Stored current prediction and image in session.")

    # Get model information
    last_trained_time, last_accuracy = model_manager.get_last_training_info()
    current_model_type = model_manager.get_current_model_type()
    model_exists = model_manager.model_exists()
    current_model_info = model_manager.get_current_model_info()
    
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
    
    return render_template('predict_new.html', 
                         prediction=display_prediction, 
                         confidence=display_confidence, 
                         all_predictions=display_all_predictions,
                         last_trained_time_obj=last_trained_time, 
                         last_trained_display=last_trained_display, 
                         last_accuracy=last_accuracy, 
                         model_exists=model_exists,
                         current_model_type=current_model_type,
                         current_model_info=current_model_info)

# --- New API Routes for Model Management ---

@app.route("/api/models", methods=['GET'])
def api_get_models():
    """API endpoint to get all models information"""
    try:
        models_info = model_manager.get_all_models_info()
        return jsonify({
            'success': True,
            'current_model': model_manager.get_current_model_type(),
            'models': models_info
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route("/api/models/switch", methods=['POST'])
def api_switch_model():
    """API endpoint to switch models"""
    try:
        data = request.get_json()
        new_model_type = data.get('model_type')
        
        if not new_model_type:
            return jsonify({'success': False, 'error': 'model_type is required'}), 400
        
        success = model_manager.switch_model(new_model_type)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'Successfully switched to {new_model_type}',
                'current_model': model_manager.get_current_model_type()
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to switch to {new_model_type}'
            }), 400
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route("/api/models/<model_type>/info", methods=['GET'])
def api_get_model_info(model_type):
    """API endpoint to get specific model information"""
    try:
        all_models_info = model_manager.get_all_models_info()
        
        if model_type not in all_models_info:
            return jsonify({'success': False, 'error': 'Model type not found'}), 404
        
        return jsonify({
            'success': True,
            'model_info': all_models_info[model_type]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route("/api/models/current", methods=['GET'])
def api_get_current_model():
    """API endpoint to get current model information"""
    try:
        return jsonify({
            'success': True,
            'current_model_type': model_manager.get_current_model_type(),
            'model_info': model_manager.get_current_model_info(),
            'training_info': {
                'last_trained': model_manager.get_last_training_info()[0].isoformat() if model_manager.get_last_training_info()[0] else None,
                'accuracy': model_manager.get_last_training_info()[1]
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == "__main__":
    print(f"[INIT] Starting Flask app with multi-model architecture on 0.0.0.0:5000...")
    app.run(debug=True, host='0.0.0.0') 