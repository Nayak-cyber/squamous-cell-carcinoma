# detect_scc.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the model once when the script is imported
model = tf.keras.models.load_model('scc_detection_model.keras')

def detect_scc(image_path, threshold=0.7):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(img_array)[0][0]
    return "SCC Detected" if prediction > threshold else "No SCC Detected"
