import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load model once
model = tf.keras.applications.MobileNetV2(weights="imagenet")

def predict_image(img):
    img = img.convert("RGB").resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    preds = model.predict(img_array)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=3)

    results = []
    for _, name, confidence in decoded[0]:
        results.append((name, round(confidence * 100, 2)))

    return results

@app.route("/", methods=["GET", "POST"])
def index():
    results = None

    if request.method == "POST":
        file = request.files.get("image")

        if file:
            img = Image.open(file.stream)
            results = predict_image(img)

    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
