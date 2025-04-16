import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Flood Segmentation App", layout="centered")

@st.cache_resource
def load_flood_model():
    return load_model("flood_save.h5")

model = load_flood_model()

st.title("🌊 Flood Area Segmentation")
st.markdown("Upload an image of a flood-affected area to detect and visualize the flooded region.")

hover_style = """
<style>
.tooltip {
  display: inline-block;
  position: relative;
}
.tooltip .tooltiptext {
  visibility: hidden;
  width: 240px;
  background-color: #444;
  color: #fff;
  text-align: center;
  padding: 6px;
  border-radius: 6px;
  position: absolute;
  z-index: 1;
  bottom: 125%;
  left: 50%;
  margin-left: -120px;
  opacity: 0;
  transition: opacity 0.3s;
}
.tooltip:hover .tooltiptext {
  visibility: visible;
  opacity: 1;
}
</style>
"""
st.markdown(hover_style, unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

tabs = st.tabs(["📷 Basic Mode", "🧠 Advanced Mode"])

with tabs[0]:  # Basic Mode
    st.markdown("""
    <div class="tooltip">
        📷 <b>Basic Mode</b> ℹ️
        <span class="tooltiptext">
        Only requires how the image was taken (e.g., phone, drone). Area is estimated with 10% margin of error.
        </span>
    </div>
    """, unsafe_allow_html=True)

    image_type = st.selectbox("Select Image Source:", ["Phone", "Drone", "Helicopter", "Satellite"])
    predict_basic = st.button("🚀 Predict Flood Area (Basic)")

    if uploaded_file and predict_basic:
        original_image = Image.open(uploaded_file).convert("RGB")
        resized_image = original_image.resize((512, 512))
        img_array = np.array(resized_image) / 255.0
        input_image = np.expand_dims(img_array, axis=0)

        with st.spinner("Segmenting image..."):
            prediction = model.predict(input_image)[0]

        binary_mask = (prediction > 0.5).astype(np.uint8)
        mask_colored = (binary_mask.squeeze() * 255).astype(np.uint8)

        white_pixels = np.sum(mask_colored == 255)
        total_pixels = mask_colored.size
        flood_percent = (white_pixels / total_pixels) * 100

        st.subheader("🖼️ Results:")
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(mask_colored, caption="Segmented Flood Area", use_container_width=True)

        st.markdown(f"### 🌧️ Estimated Flood Coverage: `{flood_percent:.2f}%`")

        # Area Calculation
        if image_type == "Phone":
            meters_per_pixel = 0.1
        elif image_type == "Drone":
            meters_per_pixel = 0.2
        elif image_type == "Helicopter":
            meters_per_pixel = 0.5
        elif image_type == "Satellite":
            meters_per_pixel = 1.0
        else:
            meters_per_pixel = 0.1

        area_per_pixel = meters_per_pixel ** 2
        flooded_area = white_pixels * area_per_pixel
        lower_bound = flooded_area * 0.90
        upper_bound = flooded_area * 1.10

        st.success(f"📏 Meters per Pixel: `{meters_per_pixel}`")
        st.success(f"🧮 Area per Pixel: `{area_per_pixel}`")
        st.success(f"⬜ White Pixels (Flooded): `{white_pixels}`")
        st.success(f"📐 Estimated Flooded Area: `{lower_bound:.2f}` m² to `{upper_bound:.2f}` m²")


with tabs[1]:  # Advanced Mode
    st.markdown("""
    <div class="tooltip">
        🧠 <b>Advanced Mode</b> ℹ️
        <span class="tooltiptext">
        Requires camera height, focal length, and sensor width. Area estimated more accurately.
        </span>
    </div>
    """, unsafe_allow_html=True)

    height = st.number_input("Enter Camera Height (in meters)", min_value=1.0, value=100.0, step=10.0)
    focal_length = st.number_input("Enter Focal Length (in mm)", min_value=1.0, value=24.0)
    sensor_width = st.number_input("Enter Sensor Width (in mm)", min_value=1.0, value=36.0)
    image_width_px = st.number_input("Enter Image Width (in pixels)", min_value=1, value=512)

    predict_advanced = st.button("🚀 Predict Flood Area (Advanced)")

    if uploaded_file and predict_advanced:
        original_image = Image.open(uploaded_file).convert("RGB")
        resized_image = original_image.resize((512, 512))
        img_array = np.array(resized_image) / 255.0
        input_image = np.expand_dims(img_array, axis=0)

        with st.spinner("Segmenting image..."):
            prediction = model.predict(input_image)[0]

        binary_mask = (prediction > 0.5).astype(np.uint8)
        mask_colored = (binary_mask.squeeze() * 255).astype(np.uint8)

        white_pixels = np.sum(mask_colored == 255)
        total_pixels = mask_colored.size
        flood_percent = (white_pixels / total_pixels) * 100

        st.subheader("🖼️ Results:")
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(mask_colored, caption="Segmented Flood Area", use_container_width=True)

        st.markdown(f"### 🌧️ Estimated Flood Coverage: `{flood_percent:.2f}%`")

        meters_per_pixel = (sensor_width * height) / (focal_length * image_width_px)
        area_per_pixel = meters_per_pixel ** 2
        flooded_area = white_pixels * area_per_pixel
        lower_bound = flooded_area * 0.95
        upper_bound = flooded_area * 1.05

        st.success(f"📏 Meters per Pixel: `{meters_per_pixel}`")
        st.success(f"🧮 Area per Pixel: `{area_per_pixel}`")
        st.success(f"⬜ White Pixels (Flooded): `{white_pixels}`")
        st.success(f"📐 Estimated Flooded Area: `{lower_bound:.2f}` m² to `{upper_bound:.2f}` m²")
