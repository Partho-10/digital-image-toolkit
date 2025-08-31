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
    processed_img = img_array.copy()
    
    st.subheader("📷 Original Image")
    st.image(img_array, caption="Original Image", width=300)  
    
    # --- Image Operations ---
    st.subheader("🎨 Image Operations")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Convert to Negative"):
            processed_img = cv2.bitwise_not(processed_img)
            st.image(processed_img, caption="Negative Image", width=300)
    
    with col2:
        if st.button("Convert to Grayscale"):
            gray = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)
            processed_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            st.image(processed_img, caption="Grayscale Image", width=300)
    
    with col3:
        if st.button("Resize to 300x300"):
            processed_img = cv2.resize(processed_img, (300, 300))
            st.image(processed_img, caption="Resized Image", width=300)
    
    # --- Sliders for adjustments ---
    st.subheader("⚙️ Adjustments")
    
    threshold_val = st.slider("Threshold Limit", 0, 255, 128)
    if st.button("Apply Threshold"):
        gray = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)
        _, thres = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
        processed_img = cv2.cvtColor(thres, cv2.COLOR_GRAY2RGB)
        st.image(processed_img, caption=f"Threshold Applied ({threshold_val})", width=300)
    
    sharp_val = st.slider("Sharpening Intensity", 1, 10, 1)
    if st.button("Apply Sharpen"):
        k = sharp_val
        kernel = np.array([[-1,-1,-1], [-1,9+k,-1], [-1,-1,-1]])
        processed_img = cv2.filter2D(processed_img, -1, kernel)
        st.image(processed_img, caption=f"Sharpen Applied ({k})", width=300)
    
    smooth_val = st.slider("Smoothing Intensity", 1, 10, 1)
    if st.button("Apply Smoothing"):
        k = smooth_val
        processed_img = cv2.GaussianBlur(processed_img, (2*k+1, 2*k+1), 0)
        st.image(processed_img, caption=f"Smoothing Applied ({k})", width=300)
    
    # --- Save / Download Image ---
    if st.button("Save Enhanced Image"):
        save_image = Image.fromarray(processed_img)
        buf = BytesIO()
        save_image.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        st.download_button(
            label="📥 Download Enhanced Image",
            data=byte_im,
            file_name="enhanced_image.jpg",
            mime="image/jpeg"
        )
