import streamlit as st
import numpy as np
from utils import load_model, read_and_preprocess, make_gradcam_heatmap, overlay_heatmap_on_image
from PIL import Image

st.set_page_config(page_title="Healthcare AI Demo")

st.title("🏥 Healthcare AI App")
st.write("Upload a Chest X-ray → Get Prediction + Grad-CAM Explanation")

MODEL_PATH = "model/model.keras"
model = load_model(MODEL_PATH)

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    arr, raw = read_and_preprocess(uploaded_file.read())

    prob = float(model.predict(arr)[0][0])
    pred_label = "PNEUMONIA" if prob >= 0.5 else "NORMAL"
    st.success(f"Prediction: {pred_label} | Probability: {prob:.4f}")

    heatmap = make_gradcam_heatmap(arr, model)
    overlay = overlay_heatmap_on_image(heatmap, raw)
    st.image([raw, overlay], caption=["Original", "Grad-CAM Overlay"], width=300)
