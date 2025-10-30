# streamlit_app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

# --- Page Config ---
st.set_page_config(page_title="MNIST Classifier", layout="centered")

# --- Custom Styling ---
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #0a1f44, #004aad);
            color: #ffffff;
        }
        .main {
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
            color: #ffffff;
        }
        h1, h2, h3, h4 {
            color: #ffffff;
            text-align: center;
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            border-radius: 10px;
            padding: 0.5em 1.5em;
            border: none;
            transition: all 0.3s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #0056b3;
            transform: scale(1.05);
            color: white;
        }
        .stMarkdown, .stFileUploader {
            color: #ffffff;
        }
        .css-1kyxreq, .css-q8sbsg, .css-1d391kg {
            color: #ffffff !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("🧠 MNIST Handwritten Digit Classifier")
st.write("Upload an image (PNG/JPG) of a handwritten digit (white on black or black on white).")

# --- Load Model Once ---
@st.cache_resource
def load_model():
    if not os.path.exists("mnist_cnn_model.h5"):
        st.error("❌ Model file 'mnist_cnn_model.h5' not found. Please train and save it first.")
        st.stop()
    model = tf.keras.models.load_model("mnist_cnn_model.h5")
    return model

model = load_model()

# --- File Upload ---
uploaded_file = st.file_uploader("📁 Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="🖼 Uploaded image", use_column_width=True)

    with st.spinner("⏳ Preprocessing and predicting..."):
        # --- Preprocess ---
        img = ImageOps.invert(image)
        img = img.resize((28, 28))
        arr = np.array(img).astype("float32") / 255.0
        arr = arr.reshape(1, 28, 28, 1)

        # --- Predict ---
        preds = model.predict(arr)
        pred = int(np.argmax(preds, axis=1)[0])
        prob = float(np.max(preds))

    st.success(f"### ✅ Predicted Digit: **{pred}**")
    st.caption(f"Confidence: **{prob:.2%}**")
    st.bar_chart(preds[0])
else:
    st.info("👉 Upload an image to get a prediction. Try cropping it to a single digit.")
