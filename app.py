# app.py
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mnist_cnn_model.h5")
    return model

model = load_model()

# App title
st.title("🧠 MNIST Digit Classifier")
st.write("Upload a 28x28 grayscale image of a handwritten digit and see the model's prediction.")

# File uploader
uploaded_file = st.file_uploader("Upload an image (png/jpg)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and preprocess image
    image = Image.open(uploaded_file).convert("L")  # grayscale
    image_resized = ImageOps.invert(image.resize((28, 28)))  # invert so white background -> black digit
    img_array = np.array(image_resized).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))  # shape (1,28,28,1)

    # Show uploaded image
    st.image(image, caption="Uploaded Image", width=150)

    # Predict
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    st.markdown(f"### 🎯 Prediction: **{predicted_label}**")
    st.markdown(f"**Confidence:** {confidence:.2f}%")
