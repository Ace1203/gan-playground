import streamlit as st
import subprocess

from inference.generate import generate_image
from inference.detect_real_fake import detect


st.title("GAN Playground")


# Model Selection

model = st.selectbox(
    "Select GAN Model",
    ["Vanilla GAN", "DCGAN", "CGAN"]
)


# CGAN Digit Selector

digit = None

if model == "CGAN":
    digit = st.slider("Select Digit to Generate", 0, 9, 3)


# Train Model

if st.button("Train Model"):

    st.write("Training started...")

    if model == "Vanilla GAN":
        subprocess.run(["python", "training/train_vanilla_gan.py"])

    elif model == "DCGAN":
        subprocess.run(["python", "training/train_dcgan.py"])

    elif model == "CGAN":
        subprocess.run(["python", "training/train_cgan.py"])

    st.success("Training completed")


# Generate Image

if st.button("Generate Image"):

    path = generate_image(model, digit)

    st.image(path)


# Upload Image for Real/Fake Detection

uploaded_file = st.file_uploader("Upload Image")

if uploaded_file is not None:

    st.image(uploaded_file)

    result = detect(uploaded_file)

    st.write("Prediction:", result)