import numpy as np
import requests
import base64
import json
import random
from PIL import Image
import io

# Load the dataset
print("Loading dataset...")
X = np.load('dataset/X.npy')
y = np.load('dataset/y.npy')

# Get a random image
random_idx = random.randint(0, len(X) - 1)
image_data = X[random_idx]
true_label = y[random_idx]

print(f"Selected image index: {random_idx}")
print(f"True label: {true_label}")
print(f"Image shape: {image_data.shape}")

# Convert numpy array to PIL Image
# Assuming the image is already in the right format (100x100 grayscale)
if image_data.ndim == 1:
    # If flattened, reshape to 100x100
    side_length = int(np.sqrt(image_data.shape[0]))
    image_data = image_data.reshape(side_length, side_length)

# Normalize to 0-255 range if needed
if image_data.max() <= 1.0:
    image_data = (image_data * 255).astype(np.uint8)
else:
    image_data = image_data.astype(np.uint8)

# Create PIL Image
img = Image.fromarray(image_data, mode='L')

# Convert to base64
buffered = io.BytesIO()
img.save(buffered, format="PNG")
img_base64 = base64.b64encode(buffered.getvalue()).decode()

# Prepare the data for the API
image_data_url = f"data:image/png;base64,{img_base64}"

# Make the prediction request
url = "http://localhost:5000/predict"
data = {
    'myImage': image_data_url,
    'action': 'predict'
}

print("\nSending prediction request...")
response = requests.post(url, data=data)

if response.status_code == 200:
    result = response.json()
    print("\n✅ Prediction successful!")
    print(f"Predicted symbol: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Current model type: {result['current_model_type']}")
    
    print("\nAll predictions:")
    if result.get('all_predictions'):
        all_preds = result['all_predictions']
        print(f"Debug - all_predictions type: {type(all_preds)}")
        print(f"Debug - all_predictions content: {all_preds}")
        
        # Handle if it's a list of dicts
        if isinstance(all_preds, list):
            for pred in all_preds:
                if isinstance(pred, dict):
                    symbol = pred.get('symbol', 'Unknown')
                    # Check if probability is nested in another dict
                    prob_data = pred.get('probability', 0)
                    if isinstance(prob_data, dict):
                        # Extract the actual probability value
                        prob_value = prob_data.get('probability', 0) if 'probability' in prob_data else 0
                    else:
                        prob_value = prob_data
                    print(f"  {symbol}: {prob_value:.4f}")
        elif isinstance(all_preds, dict):
            # Sort by probability
            items = []
            for symbol, prob_data in all_preds.items():
                if isinstance(prob_data, dict):
                    # Extract probability from nested dict
                    prob_value = prob_data.get('probability', 0)
                else:
                    prob_value = prob_data
                items.append((symbol, prob_value))
            
            sorted_preds = sorted(items, key=lambda x: x[1], reverse=True)
            for symbol, prob in sorted_preds:
                print(f"  {symbol}: {prob:.4f}")
else:
    print(f"\n❌ Error: {response.status_code}")
    print(response.text)

# Map numeric labels to symbols for comparison
symbols_map = {0: 'α', 1: 'β', 2: 'ε'}
print(f"\n📊 True label: {symbols_map.get(true_label, f'Unknown ({true_label})')}") 