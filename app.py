from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from io import BytesIO
from PIL import Image
import requests

app = Flask(__name__)
CORS(app)

endpoint = "http://localhost:8501/v1/models/potatoes_model:predict"
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy", "Undefined"]

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"message": "Hello, I am alive"})

def read_file_as_image(data) -> np.ndarray:
    try:
        image = np.array(Image.open(BytesIO(data)))
    except Exception:
        try:
            image = np.array(Image.open(BytesIO(bytes(data))))
        except Exception:
            return None
    return image

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"detail": "No file provided"}), 400

    file = request.files['file']
    image = read_file_as_image(file.read())

    if image is None:
        return jsonify({"detail": "Invalid image data"}), 400

    img_batch = np.expand_dims(image, 0)

    json_data = {
        "instances": img_batch.tolist()
    }

    response = requests.post(endpoint, json=json_data)
    if response.status_code != 200:
        return jsonify({"detail": "Error in model prediction"}), response.status_code

    prediction = np.array(response.json().get("predictions", [])[0])
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    return jsonify({
        "class": predicted_class,
        "confidence": float(confidence)
    })

if __name__ == "__main__":
    #app.run(host='0.0.0.0', port=8000)
    app.run(debug=True)
