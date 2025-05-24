import os

# Get the absolute path of the directory where this config.py file is located.
# This will be the 'predecir_simbolos' directory.
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Symbol configuration
SYMBOLS = ['alpha', 'beta', 'epsilon'] # α, β, ε
SYMBOLS_DISPLAY = {'alpha': 'α', 'beta': 'β', 'epsilon': 'ε'}

# Image processing configuration
IMAGE_DIM = (100, 100) # Standard image dimension for the model

# Files and Directories
# Define paths relative to APP_DIR for robustness
DATASET_BASE_PATH = os.path.join(APP_DIR, 'dataset')
MODEL_BASE_PATH = os.path.join(APP_DIR, 'model')
DB_FOLDER = os.path.join(APP_DIR, 'db')
STATIC_FOLDER = os.path.join(APP_DIR, 'static')
TEMPLATES_FOLDER = os.path.join(APP_DIR, 'templates')


MODEL_FILENAME = os.path.join(MODEL_BASE_PATH, 'modelo_simbolos.pkl')
LAST_TRAINING_FILE = os.path.join(DB_FOLDER, 'last_training_time.txt')

# Output .npy files for dataset export
X_NPY_PATH = os.path.join(DATASET_BASE_PATH, 'X.npy')
Y_NPY_PATH = os.path.join(DATASET_BASE_PATH, 'y.npy')

# Training configuration
MIN_TRAINING_INTERVAL = 5  # Minimum minutes between trainings 