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
        
        # Handle dict format: {'alpha': {'probability': 0.02, 'symbol': 'α'}, ...}
        if isinstance(all_preds, dict):
            # Extract probabilities and sort
            prob_list = []
            for key, value in all_preds.items():
                if isinstance(value, dict):
                    symbol = value.get('symbol', key)
                    prob = value.get('probability', 0)
                    prob_list.append((symbol, prob))
            
            # Sort by probability descending
            prob_list.sort(key=lambda x: x[1], reverse=True)
            
            for symbol, prob in prob_list:
                print(f"  {symbol}: {prob:.4f}")
                
else:
    print(f"\n❌ Error: {response.status_code}")
    print(response.text)

# Map numeric labels to symbols for comparison
symbols_map = {'alpha': 'α', 'beta': 'β', 'epsilon': 'ε'}
true_symbol = symbols_map.get(true_label, true_label)
print(f"\n📊 Comparison:")
print(f"True label: {true_symbol}")
print(f"Predicted: {result.get('prediction', 'Unknown') if response.status_code == 200 else 'Error'}")
print(f"Correct: {'✅' if response.status_code == 200 and result.get('prediction') == true_symbol else '❌'}") 