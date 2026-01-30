import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

# === 1. SMART PATHING ===
# This finds the 'face_emotionModel.h5' inside the 'model' folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "face_emotionModel.h5")

# === 2. LOAD MODEL ===
@st.cache_resource
def load_my_model():
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at: {model_path}")
        return None
    return load_model(model_path)

model = load_my_model()

# Emotion labels matching your model's 7 classes
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# === 3. STREAMLIT GUI ===
st.title("🎭 Face Emotion Detector")
st.write("Upload a face image to predict the emotion.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # === 4. PREPROCESSING ===
    # Convert to grayscale and resize to 48x48 as required by your model
    img = image.convert('L')
    img = img.resize((48, 48))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Rescale to [0, 1]

    # === 5. PREDICTION ===
    prediction = model.predict(img_array)
    max_index = np.argmax(prediction[0])
    predicted_emotion = EMOTIONS[max_index]
    confidence = prediction[0][max_index] * 100

    st.success(f"**Prediction: {predicted_emotion}** ({confidence:.2f}% confidence)")