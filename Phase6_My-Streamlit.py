#-----------------------------------------------------------------------------------------------
#   Capstone Project 6 — Healthcare: Pneumonia Detection from Chest X-rays [ Tony Hevey]
#
#
# Purpose: Phase 6 — Model Deployment & Streamlit App
#
#   Load the best CNN model and deploy it as a Streamlit app for pneumonia detection from chest X-ray images.
#
#   Output: Picture of the uploaded X-ray image, prediction result (NORMAL or PNEUMONIA), and confidence score.
#



import streamlit as st
import os
import tensorflow as tf
import numpy as np
from PIL import Image


CLASSES = ["NORMAL", "PNEUMONIA"]
BEST_CNN_MODEL = os.path.join(os.path.dirname(__file__), "best_cnn_model.keras")

model = tf.keras.models.load_model(BEST_CNN_MODEL)

st.title("XRay Image - Pneumonia Detection")

st.write("Upload an XRay image to analyze for pneumonia.")

uploaded_file = st.file_uploader("Choose an XRay image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded XRay Image for Analysis', use_container_width=True)

    input_shape = model.input_shape

    # Prepare the image for prediction - Make it as the model expects.
    st.write("Model input shape:", input_shape)
    height = int(input_shape[1])
    width = int(input_shape[2] )
    channels = int(input_shape[3])
    mode = "L"

    img = Image.open(uploaded_file).convert(mode).resize((width, height))
    array = np.asarray(img, dtype=np.float32) / 255.0

    tensor = np.expand_dims(array, axis=0)
    probabilities = model.predict(tensor, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    result = CLASSES[predicted_index]
    confidence = float(probabilities[predicted_index]) * 100

    st.write("Analyzing the image...")
    st.write(f"Result: **{result}** ({confidence:.1f}% confidence)")
