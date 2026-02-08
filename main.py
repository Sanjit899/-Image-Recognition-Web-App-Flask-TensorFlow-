import tensorflow as tf
import numpy as np
from PIL import Image

# Load pre-trained MobileNet model
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Load and preprocess image
img = Image.open("test.png").resize((224, 224))
img_array = np.array(img)

# Expand dimensions to match model input
img_array = np.expand_dims(img_array, axis=0)

# Preprocess image
img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

# Make prediction
predictions = model.predict(img_array)

# Decode predictions
decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)

# Print results
for i, (imagenet_id, label, confidence) in enumerate(decoded[0]):
    print(f"{i+1}. {label} ({confidence*100:.2f}%)")
