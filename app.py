import streamlit as st
import numpy as np
import cv2
from PIL import Image
from io import BytesIO

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(page_title="✨ Digital Image Toolkit ✨", layout="wide")
st.title("✨ Digital Image Toolkit ✨")

# --- Sidebar Profile ---
with st.sidebar:
    st.header("👤 Profile")
    # Make sure you have the "Partho.JPG" image in the same folder as your script
    try:
        st.image("Partho.JPG", width=120)
    except FileNotFoundError:
        st.warning("Profile image (Partho.JPG) not found.")
    st.text("Name: Partho Sarkar")
    st.text("ID: 0812220205101018")

# --- Image Upload ---
uploaded_file = st.file_uploader("Select an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    img = Image.open(uploaded_file)
    original_img_array = np.array(img.convert("RGB"))

    # Initialize session state only if it's the first run or a new image is uploaded
    # We use the file's unique file_id to check if it's a new upload
    if "processed_img" not in st.session_state or "last_image_id" not in st.session_state or st.session_state.last_image_id != uploaded_file.file_id:
        st.session_state.processed_img = original_img_array.copy()
        st.session_state.last_image_id = uploaded_file.file_id 

    # --- Display Original + Processed side by side ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Original Image")
        st.image(original_img_array, caption="Original Image", use_container_width=True)

    with col2:
        st.subheader("✨ Processed Image")
        st.image(st.session_state.processed_img, caption="Processed Image", use_container_width=True)

    # --- Image Operations ---
    st.subheader("🎨 Image Operations")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        if st.button("Convert to Negative", key="btn_neg", use_container_width=True):
            st.session_state.processed_img = cv2.bitwise_not(st.session_state.processed_img)
            st.rerun()

    with c2:
        if st.button("Convert to Grayscale", key="btn_gray", use_container_width=True):
            if len(st.session_state.processed_img.shape) == 3:
                gray = cv2.cvtColor(st.session_state.processed_img, cv2.COLOR_RGB2GRAY)
                st.session_state.processed_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
                st.rerun()

    with c3:
        if st.button("Resize to 300x300", key="btn_resize", use_container_width=True):
            st.session_state.processed_img = cv2.resize(st.session_state.processed_img, (300, 300))
            st.rerun()

    with c4:
        if st.button("🔄 Reset Image", key="btn_reset", use_container_width=True):
            st.session_state.processed_img = original_img_array.copy()
            st.rerun()

    # --- Sliders for adjustments ---
    st.subheader("⚙️ Adjustments")

    threshold_val = st.slider("Threshold Limit", 0, 255, 128)
    if st.button("Apply Threshold", key="btn_thresh"):
        gray = cv2.cvtColor(st.session_state.processed_img, cv2.COLOR_RGB2GRAY)
        _, thres = cv2.threshold(gray, threshold_val, 255, cv2.THRESH_BINARY)
        st.session_state.processed_img = cv2.cvtColor(thres, cv2.COLOR_GRAY2RGB)
        st.rerun()

    sharp_val = st.slider("Sharpening Intensity", 0, 10, 1)
    if st.button("Apply Sharpen", key="btn_sharp"):
        k = sharp_val
        kernel = np.array([[-1, -1, -1],
                           [-1, 9 + k, -1],
                           [-1, -1, -1]])
        st.session_state.processed_img = cv2.filter2D(st.session_state.processed_img, -1, kernel)
        st.rerun()

    smooth_val = st.slider("Smoothing Intensity", 1, 15, 1)
    if st.button("Apply Smoothing", key="btn_smooth"):
        k_size = 2 * smooth_val + 1
        st.session_state.processed_img = cv2.GaussianBlur(st.session_state.processed_img, (k_size, k_size), 0)
        st.rerun()

    # --- Save / Download Image ---
    st.subheader("💾 Save Image")

    processed_image_pil = Image.fromarray(st.session_state.processed_img)
    buf = BytesIO()
    processed_image_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    st.download_button(
        label="📥 Download Processed Image",
        data=byte_im,
        file_name="processed_image.png",
        mime="image/png",
        use_container_width=True
    )
else:
    st.info("👋 Welcome! Please upload an image to get started.")