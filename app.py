from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os	
from huggingface_hub import hf_hub_download


model_path = hf_hub_download("Adarsh921/ecodetect-resnet50", "waste_classifier.keras")
model = load_model(model_path)

app = Flask(__name__, template_folder='templates')

# Preprocess function
def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")  
    img = img.resize((224,224))
    img_array = np.array(img)
    img_array = img_array / 255                      
    img_array = np.expand_dims(img_array, axis=0)           
    return img_array


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['file']
    file_path = "temp.jpg"
    file.save(file_path)

    img = preprocess_image(file_path)
    prediction = model.predict(img)[0][0]

    os.remove(file_path)

    # Confidence score and label
    confidence = float(prediction) if prediction > 0.5 else 1 - float(prediction)
    label = "Non-Biodegradable" if prediction >= 0.5 else "Biodegradable"

    return jsonify({
        'prediction': label,
        'confidence': f"{confidence * 100:.2f}%"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)