import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.title("🚦 Traffic Sign Recognition")

uploaded_file = st.file_uploader("Upload a traffic sign image", type=["png", "jpg", "jpeg"])
model = load_model("model/model.keras")

# Load class names
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f]

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((64, 64))
    st.image(image, caption="Uploaded Image", use_container_width=True)  
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"🔍 Predicted Sign: **{predicted_class}**")
