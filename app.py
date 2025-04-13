import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# 🟢 Ensure this is the first Streamlit command
st.set_page_config(page_title="Flood Segmentation App", layout="centered")

# 🔁 Cache the model so it's loaded only once
@st.cache_resource
def load_flood_model():
    return load_model("flood_save.h5")

model = load_flood_model()

st.title("🌊 Flood Area Segmentation")
st.markdown("Upload an image of a flood-affected area to detect and visualize the flooded region.")

# 📤 Image uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and resize image to match model input (512x512)
    original_image = Image.open(uploaded_file).convert("RGB")
    resized_image = original_image.resize((512, 512))
    img_array = np.array(resized_image) / 255.0  # Normalize
    input_image = np.expand_dims(img_array, axis=0)

    # 🌀 Show spinner during prediction
    with st.spinner("Segmenting image... Please wait ⏳"):
        prediction = model.predict(input_image)[0]
    
    # ⬛ Convert prediction to binary mask (threshold)
    binary_mask = (prediction > 0.5).astype(np.uint8)  # 1 for flooded, 0 for non-flooded

    # 🎨 Convert binary mask to a colored image (white flooded, black non-flooded)
    mask_colored = (binary_mask.squeeze() * 255).astype(np.uint8)

    # 🧮 Calculate flooded area percentage
    white_pixels = np.sum(mask_colored == 255)
    total_pixels = mask_colored.size
    flood_percent = (white_pixels / total_pixels) * 100

    # 🖼️ Show original + predicted side by side
    st.subheader("🖼️ Results:")
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Original Image", use_container_width=True)
    with col2:
        st.image(mask_colored, caption="Segmented Flood Area", use_container_width=True)

    # 📊 Show flood coverage
    st.markdown(f"### 🌧️ Estimated Flood Coverage: `{flood_percent:.2f}%`")

