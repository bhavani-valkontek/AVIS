# scratch_module.py

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import io
import time
from PIL import Image
from datetime import datetime

# Replace these imports with your actual functions or implementations
from your_model_module import predict_image, create_mask_overlay


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

            if masks:
                mask_img, overlay_img, severity, total_pixels, confidence = create_mask_overlay(
                    original_np, masks, scores
                )

                st.subheader("Results:")
                st.write(f"🕒 Detection Time: {detection_time:.2f} seconds")

                col1, col2 = st.columns(2)
                col1.image(mask_img, caption="🩹 Scratch Mask", use_container_width=True)
                col2.image(overlay_img, caption="🧪 Result Overlay", use_container_width=True)

                st.subheader("Scratch Info:")
                st.write(f"📉 Severity: `{severity:.1f}%`")
                st.write(f"📏 Pixels Affected: `{total_pixels}px`")
                st.write(f"🎯 Confidence: `{confidence:.1f}%`")

                # Save images to buffer
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

                # Save result as CSV
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
