from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model at startup
MODEL = tf.keras.models.load_model("../saved_models/3")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy", "Undefined"]

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "Hello, I am alive"})

def read_file_as_image(data) -> np.ndarray:
    try:
        image = np.array(Image.open(BytesIO(data)))
    except Exception:
        return None
    return image

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files and 'blob_file' not in request.form:
        return jsonify({"detail": "No file provided"}), 400

    if 'file' in request.files:
        file = request.files['file']
        image = read_file_as_image(file.read())
    elif 'blob_file' in request.form:
        blob_file = request.form['blob_file']
        image = read_file_as_image(blob_file.encode())

    if image is None:
        return jsonify({"detail": "Invalid image data"}), 400

    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return jsonify({
        'class': predicted_class,
        'confidence': float(confidence)
    })

if __name__ == "__main__":
    app.run(host='localhost', port=8000)
