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