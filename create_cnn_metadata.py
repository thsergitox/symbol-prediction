import joblib
import os

# Create metadata for the TensorFlow CNN model
model_data = {
    'model_path_h5': './model/mejor_modelo.h5',
    'symbols_display': {'alpha': 'α', 'beta': 'β', 'epsilon': 'ε'},
    'training_accuracy': 0.95,  # Placeholder - will be loaded from actual model
    'classes': ['alpha', 'beta', 'epsilon'],
    'preprocessor_config': {'target_size': (100, 100)},
    'model_type': 'tensorflow_cnn'
}

# Save metadata as .pkl
pkl_path = './model/modelo_tensorflow_cnn.pkl'
os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
joblib.dump(model_data, pkl_path)

print(f"CNN metadata saved to {pkl_path}")
print("Model data:", model_data) 