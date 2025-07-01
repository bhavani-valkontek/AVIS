
# dent_module.py
from distutils.command.build import build

import streamlit as st
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import json
from PIL import Image
import os
import requests
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials

# ===============================
# 🎯 Download model from Hugging Face
# ===============================
def download_model_from_huggingface(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        return save_path
    else:
        raise Exception(f"Failed to download model from Hugging Face. Status code: {response.status_code}")

# ===============================
# 🎯 Function: Run YOLO Inference
# ===============================
def run_inference(image_path, model_path, conf_threshold):
    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=conf_threshold, imgsz=640, save=False)
    output = results[0]

    image = cv2.imread(image_path)
    image_draw = image.copy()
    detection_data = []

    if output.boxes is not None and len(output.boxes) > 0:
        for box in output.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box
            class_name = output.names[int(cls)]

            # Draw bounding box
            cv2.rectangle(image_draw, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(image_draw, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            detection_data.append({
                "class": class_name,
                "confidence": float(f"{conf:.4f}"),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
            })
    return image_draw, detection_data

# ====================================
# ☁️ Upload File to Google Drive API using Service Account
# ====================================
# ☁️ Upload File to Google Drive (Service Account)
# ====================================
def upload_to_drive(filepath, filename, folder_id=None):
    SCOPES = ['https://www.googleapis.com/auth/drive.file']

    # ✅ Load credentials from Streamlit secrets
    credentials_info = st.secrets["GDRIVE_SERVICE_ACCOUNT"]
    creds = service_account.Credentials.from_service_account_info(credentials_info, scopes=SCOPES)

    service = build('drive', 'v3', credentials=creds)

    file_metadata = {'name': filename}
    if folder_id:
        file_metadata['parents'] = [folder_id]

    media = MediaFileUpload(filepath, mimetype='image/jpeg')
    uploaded = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()

    return uploaded.get('id')

# ============================
# 🚀 Streamlit Web Application
# ============================
def dent_ui():
    st.title("🔍Vehicle Dent Detection")
    st.markdown("Upload an image or use your camera to detect car dents.")

    # Hugging Face model URL (📌 update this to your actual Hugging Face URL)
    hf_model_url = "https://huggingface.co/babbilibhavani/scartch_detection/resolve/main/best_model.pt"

    # Load model once
    with st.spinner("📦 Downloading model from Hugging Face..."):
        try:
            model_file = download_model_from_huggingface(hf_model_url, "../best_model.pt")
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            st.stop()

    # Upload image or capture
    image_file = st.file_uploader("🖼️ Upload Image", type=["jpg", "jpeg", "png"])
    # camera_image = st.camera_input("📷 Or take a photo")
    conf_threshold = st.slider("🎯 Confidence Threshold", 0.05, 1.0, 0.25, 0.05)

    final_image = None
    if image_file is not None:
        final_image = Image.open(image_file).convert("RGB")
    # elif camera_image is not None:
    #     final_image = Image.open(camera_image).convert("RGB")

    if final_image:
        with st.spinner("⏳ Running Detection..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
                final_image.save(tmp_img.name)
                tmp_img_path = tmp_img.name

            output_image, detections = run_inference(tmp_img_path, model_file, conf_threshold)
            st.subheader("📊 Detection Results")

            # Show image
            st.image(output_image, caption="🖼️ Detected Image", channels="BGR", use_container_width=True)

            # Save output image
            output_image_path = "dent_detection_output.jpg"
            cv2.imwrite(output_image_path, output_image)

            if detections:
                # ✅ Save JSON
                json_path = "detection_results.json"
                with open(json_path, "w") as f:
                    json.dump(detections, f, indent=2)

                # 💾 Download buttons
                st.download_button("⬇️ Download Results (JSON)", data=json.dumps(detections, indent=2),
                                   file_name="detection_results.json", mime="application/json")
                with open(output_image_path, "rb") as img_file:
                    st.download_button("⬇️ Download Output Image", img_file.read(),
                                       file_name="dent_detection_output.jpg", mime="image/jpeg")

                # ☁️ Upload to Google Drive
                try:
                    folder_id = "12fhgEhNBRxx560dmBLpB64fbQJ3-lqWd"  # Replace with your actual folder ID
                    drive_file_id = upload_to_drive(output_image_path, "dent_detection_output.jpg", folder_id)
                    st.success(f"✅ Output image uploaded to Google Drive.")
                except Exception as e:
                    st.error(f"❌ Failed to upload to Google Drive: {e}")
            else:
                st.warning("❌ No dents detected in the image.")




