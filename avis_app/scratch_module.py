import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import io
import time
from datetime import datetime
import torch
from torchvision import transforms
import cv2

# ==========================
# ⚙️ Model Loading
# ==========================
@st.cache_resource
def load_model():
    model = torch.load("best_scratch_model.pt", map_location="cpu")  # or 'cuda' if using GPU
    model.eval()
    return model

model = load_model()

# ==========================
# 🔍 Prediction Function
# ==========================
def predict_image(image: Image.Image):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0)  # [1, C, H, W]

    with torch.no_grad():
        prediction = model(img_tensor)[0]

    masks = prediction.get("masks", None)
    scores = prediction.get("scores", None)

    if masks is not None and scores is not None:
        # Convert masks to NumPy and filter by score threshold
        masks = masks.squeeze().cpu().numpy()
        scores = scores.cpu().numpy()
        keep = scores > 0.5  # confidence threshold
        masks = masks[keep]
        scores = scores[keep]
    else:
        masks, scores = [], []

    original_np = np.array(image)

    return original_np, masks, scores

# ==========================
# 🎨 Create Overlay and Stats
# ==========================
def create_mask_overlay(original_np, masks, scores):
    total_pixels = 0
    all_masks = np.zeros(original_np.shape[:2], dtype=np.uint8)

    for mask in masks:
        binary_mask = (mask > 0.5).astype(np.uint8)
        all_masks += binary_mask
        total_pixels += np.sum(binary_mask)

    severity = (total_pixels / (original_np.shape[0] * original_np.shape[1])) * 100
    confidence = np.mean(scores) * 100

    # Create mask image (in red)
    mask_img = original_np.copy()
    mask_img[all_masks > 0] = [255, 0, 0]  # red scratches
    mask_img = Image.fromarray(mask_img)

    # Create overlay image
    overlay_img = original_np.copy()
    overlay = np.zeros_like(overlay_img)
    overlay[all_masks > 0] = [255, 0, 0]  # red overlay
    overlay_img = cv2.addWeighted(overlay_img, 0.7, overlay, 0.3, 0)
    overlay_img = Image.fromarray(overlay_img)

    return mask_img, overlay_img, severity, total_pixels, confidence

# ==========================
# 🌟 Streamlit UI Function
# ==========================
def scratch_ui():
    st.title("🚗 Vehicle Scratch Detection")

    uploaded_file = st.file_uploader("Upload vehicle image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        st.image(image, caption="Original", use_container_width=True)

        st.download_button(
            label="⬇️ Download Original Image",
            data=image_data,
            file_name="original_image.jpg",
            mime="image/jpeg"
        )

        if st.button("Run Detection"):
            with st.spinner("🔍 Detecting Scratches..."):
                start_time = time.time()
                original_np, masks, scores = predict_image(image)
                end_time = time.time()
                detection_time = end_time - start_time

            if len(masks) > 0:
                mask_img, overlay_img, severity, total_pixels, confidence = create_mask_overlay(original_np, masks, scores)

                st.subheader("Results:")
                st.write(f"🕒 Detection Time: {detection_time:.2f} seconds")

                col1, col2 = st.columns(2)
                col1.image(mask_img, caption="🩹 Scratch Mask", use_container_width=True)
                col2.image(overlay_img, caption="🧪 Result Overlay", use_container_width=True)

                st.subheader("Scratch Info:")
                st.write(f"📉 Severity: `{severity:.1f}%`")
                st.write(f"📏 Pixels Affected: `{total_pixels}px`")
                st.write(f"🎯 Confidence: `{confidence:.1f}%`")

                # Prepare downloadable images
                mask_buffer = io.BytesIO()
                overlay_buffer = io.BytesIO()
                mask_img.save(mask_buffer, format="JPEG")
                overlay_img.save(overlay_buffer, format="JPEG")

                st.download_button(
                    label="⬇️ Download Mask Image",
                    data=mask_buffer.getvalue(),
                    file_name="scratch_mask.jpg",
                    mime="image/jpeg"
                )

                st.download_button(
                    label="⬇️ Download Overlay Image",
                    data=overlay_buffer.getvalue(),
                    file_name="scratch_overlay.jpg",
                    mime="image/jpeg"
                )

                # CSV result data
                csv_data = pd.DataFrame([{
                    "severity": severity,
                    "confidence": confidence,
                    "pixels": total_pixels,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }])
                csv_buffer = io.StringIO()
                csv_data.to_csv(csv_buffer, index=False)

                st.download_button(
                    label="⬇️ Download Result as CSV",
                    data=csv_buffer.getvalue(),
                    file_name="scratch_detection_results.csv",
                    mime="text/csv"
                )

                st.success("✅ Detection Completed Successfully.")
            else:
                st.warning("⚠️ No scratches detected.")
