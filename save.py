import os
import base64
import glob
import numpy as np
from flask import request, redirect, send_file
from skimage import io, transform
from datetime import datetime

# Import configurations from config.py
from config import (
    SYMBOLS, SYMBOLS_DISPLAY, IMAGE_DIM, DATASET_BASE_PATH, 
    X_NPY_PATH, Y_NPY_PATH
)

def prepare_symbol_dataset_files(): # Back to original name
    print("[DATASET_EXPORT] Starting generation of X.npy and y.npy...")
    images_data = []
    labels_data = []
    
    for symbol_name in SYMBOLS:
        print(f"[DATASET_EXPORT] Processing images for symbol '{SYMBOLS_DISPLAY.get(symbol_name, symbol_name)}' ({symbol_name})...")
        symbol_image_folder = os.path.join(DATASET_BASE_PATH, symbol_name)
        
        if not os.path.isdir(symbol_image_folder):
            print(f"[WARN] Directory not found for symbol '{symbol_name}': {symbol_image_folder}")
            continue

        filelist = glob.glob(os.path.join(symbol_image_folder, '*.png'))
        print(f"[DATASET_EXPORT] Found {len(filelist)} images in {symbol_image_folder}/")
        
        if not filelist:
            print(f"[WARN] No images found in folder for symbol '{symbol_name}'.")
            continue
            
        for img_path in filelist:
            try:
                img = io.imread(img_path)
                processed_img = None
                if img.ndim == 3 and img.shape[2] == 4: # RGBA
                    processed_img = img[:, :, 3] # Use alpha channel
                elif img.ndim == 3 and img.shape[2] == 3: # RGB
                    processed_img = np.mean(img, axis=2).astype(np.uint8) # Convert to grayscale
                elif img.ndim == 2: # Grayscale
                    processed_img = img
                else:
                    print(f"[ERROR] Unsupported image format or dimension for {img_path}: {img.shape}, ndim={img.ndim}")
                    continue
                
                img_resized = transform.resize(processed_img, IMAGE_DIM,
                                             anti_aliasing=True, 
                                             preserve_range=True).astype(np.uint8)
                images_data.append(img_resized)
                labels_data.append(symbol_name)
            except Exception as e:
                print(f"[ERROR] Error loading or processing image {img_path}: {e}")

    if not images_data:
        print("[DATASET_EXPORT] No images were processed. X.npy and y.npy will be based on empty arrays.")
        X_np_array = np.array([])
        y_np_array = np.array([])
    else:
        X_np_array = np.array(images_data)
        y_np_array = np.array(labels_data)
    
    print(f"[DATASET_EXPORT] Final X_data shape: {X_np_array.shape}, y_data shape: {y_np_array.shape}")

    os.makedirs(DATASET_BASE_PATH, exist_ok=True) # Ensure dataset directory exists
    
    try:
        np.save(X_NPY_PATH, X_np_array)
        print(f"[DATASET_EXPORT] Saved X.npy to {X_NPY_PATH}")
        np.save(Y_NPY_PATH, y_np_array)
        print(f"[DATASET_EXPORT] Saved y.npy to {Y_NPY_PATH}")
        return X_NPY_PATH, Y_NPY_PATH
    except Exception as e:
        print(f"[ERROR] Failed to save .npy files: {e}")
        return None, None

def upload_image():
    print("[ROUTE] POST /upload-image")
    try:
        img_data = request.form.get('myImage').replace("data:image/png;base64,", "")
        symbol_to_draw = request.form.get('symbol')
        print(f"[UPLOAD] Received image for symbol: {symbol_to_draw}")
        if not symbol_to_draw or symbol_to_draw not in SYMBOLS:
            print(f"[ERROR] Invalid symbol specified: {symbol_to_draw}")
            return "Invalid symbol specified.", 400
        
        symbol_folder = os.path.join(DATASET_BASE_PATH, symbol_to_draw)
        os.makedirs(symbol_folder, exist_ok=True) # Ensure directory exists
        
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

def download_X_symbols():
    print("[ROUTE] GET /X.npy (Symbol dataset)")
    x_path, _ = prepare_symbol_dataset_files()
    if x_path and os.path.exists(x_path):
        return send_file(x_path, as_attachment=True, download_name='X.npy')
    else:
        print("[ERROR] X.npy could not be generated or found.")
        return "Error generating or finding X.npy. Check server logs.", 500

def download_y_symbols():
    print("[ROUTE] GET /y.npy (Symbol dataset)")
    _, y_path = prepare_symbol_dataset_files()
    if y_path and os.path.exists(y_path):
        return send_file(y_path, as_attachment=True, download_name='y.npy')
    else:
        print("[ERROR] y.npy could not be generated or found.")
        return "Error generating or finding y.npy. Check server logs.", 500

# Router function - similar to FastAPI's APIRouter
def register_routes(app):
    """
    Register all save-related routes to the Flask app.
    This acts like a router, similar to FastAPI's APIRouter pattern.
    """
    print("[ROUTER] Registering save routes...")
    
    # Register upload route
    app.add_url_rule('/upload-image', 'upload_image', upload_image, methods=['POST'])
    print("[ROUTER] Registered POST /upload-image -> upload_image")
    
    # Register download routes
    app.add_url_rule('/X.npy', 'download_X_symbols', download_X_symbols, methods=['GET'])
    print("[ROUTER] Registered GET /X.npy -> download_X_symbols")
    
    app.add_url_rule('/y.npy', 'download_y_symbols', download_y_symbols, methods=['GET'])
    print("[ROUTER] Registered GET /y.npy -> download_y_symbols")
    
    print("[ROUTER] All save routes registered successfully!") 