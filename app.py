import streamlit as st
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

st.set_page_config(page_title="✨ Digital Image Toolkit ✨", layout="wide")
st.title("✨ Digital Image Toolkit ✨")

# --- Sidebar Profile ---
st.sidebar.header("👤 Profile")
st.sidebar.image("Partho.JPG", width=120)
st.sidebar.text("Name: Partho Sarkar")
st.sidebar.text("ID: 0812220205101018")

# --- Image Upload ---
uploaded_file = st.file_uploader("Select an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img_array = np.array(img.convert("RGB"))

    # Initialize session state for processed image
    if "processed_img" not in st.session_state:
        st.session_state.processed_img = img_array.copy()

    # --- Show Original + Processed side by side ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Original Image")
        st.image(img_array, caption="Original Image", width=300)

    with col2:
        st.subheader("✨ Processed Image")
        st.image(st.session_state.processed_img, caption="Enhanced Image", width=300)

    # --- Image Operations ---
    st.subheader("🎨 Image Operations")
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        if st.button("Convert to Negative", key="btn_neg"):
            st.session_state.processed_img = cv2.bitwise_not(st.session_state.processed_img)
    
    with c2:
        if st.button("Convert to Grayscale", key="btn_gray"):
            gray = cv2.cvtColor(st.session_state.processed_img, cv2.COLOR_RGB2GRAY)
            st.session_state.processed_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    with c3:
        if st.button("Resize to 300x300", key="btn_resize"):
            st.session_state.processed_img = cv2.resize(st.session_state.processed_img, (300, 300))

    with c4:
        if st.button("🔄 Reset Image", key="btn_reset"):
            st.session_state.processed_img = img_array.copy()
    
    # --- Sliders for adjustments ---
    st.subheader("⚙️ Adjustments")
    
    threshold_val = st.slider("Threshold Limit", 0, 255, 128)
    if st.button("Apply Threshold", key="btn_thresh"):
        gray = cv2.cvtColor(st.session_state.processed_img, cv2.COLOR_RGB2GRAY)
        _, thres = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
        st.session_state.processed_img = cv2.cvtColor(thres, cv2.COLOR_GRAY2RGB)
    
    sharp_val = st.slider("Sharpening Intensity", 1, 10, 1)
    if st.button("Apply Sharpen", key="btn_sharp"):
        k = sharp_val
        kernel = np.array([[-1,-1,-1], [-1,9+k,-1], [-1,-1,-1]])
        st.session_state.processed_img = cv2.filter2D(st.session_state.processed_img, -1, kernel)
    
    smooth_val = st.slider("Smoothing Intensity", 1, 10, 1)
    if st.button("Apply Smoothing", key="btn_smooth"):
        k = smooth_val
        st.session_state.processed_img = cv2.GaussianBlur(st.session_state.processed_img, (2*k+1, 2*k+1), 0)

    # --- Save / Download Image ---
    st.subheader("💾 Save Image")
    if st.button("Save Enhanced Image", key="btn_save"):
        save_image = Image.fromarray(st.session_state.processed_img)
        buf = BytesIO()
        save_image.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        st.download_button(
            label="📥 Download Enhanced Image",
            data=byte_im,
            file_name="enhanced_image.jpg",
            mime="image/jpeg"
        )
