import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

model = tf.keras.models.load_model("rps_model.h5")

st.title("✊✋✌ Rock Paper Scissors Classifier")

uploaded_file = st.file_uploader(
    "Upload a Rock / Paper / Scissors image",
    type=["jpg", "png"]
)

classes = ["Rock", "Paper", "Scissors"]

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB").resize((128, 128))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    st.image(image, caption="Uploaded Image")
    st.success(f"Predicted Gesture: {classes[class_index]}")

