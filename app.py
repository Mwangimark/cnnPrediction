from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import numpy as np
import cv2  # OpenCV for handling image data
import os

app = Flask(__name__)

# Load the saved model (adjust the path if necessary)
model = load_model("cnn_model.h5")  # Or "cnn_model" if using the SavedModel format
print("Model loaded successfully!")

# Define a route for the home page (HTML form to upload image)
@app.route('/')
def home():
    return render_template('index.html')

# Define a route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the form
        image_file = request.files['image']
        
        # Preprocess the image before prediction
        img = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))  # Resize to the model's input size (28x28 for MNIST)
        img = img.astype('float32') / 255  # Normalize pixel values
        img = np.expand_dims(img, axis=-1)  # Add channel dimension (28, 28, 1)
        img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 28, 28, 1)
        
        # Make a prediction
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)[0]
        
        # Return the prediction as JSON
        return jsonify({'predicted_class': int(predicted_class)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

