import streamlit as st
from ultralytics import YOLO
import cv2
import os
import tempfile
from PIL import Image
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials
import time
import requests

# =============================
# CONFIGURATION
# =============================
MODEL_URL = "https://huggingface.co/babbilibhavani/scartch_detection/resolve/main/best_crk.pt"
DRIVE_FOLDER_ID = "1xqOTdWI3-9uhNr_tBV2fbDk4rqXP0O76"  # 🔁 Replace this with your actual Drive folder ID
JSON_PATH =st.secrets["GDRIVE_SERVICE_ACCOUNT"]
CONFIDENCE_DEFAULT = 0.3


# =============================
# AUTHENTICATE GOOGLE DRIVE
# =============================
@st.cache_resource
def authenticate_drive(_json_path):
    gauth = GoogleAuth()
    scope = ['https://www.googleapis.com/auth/drive']
    gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name(json_path, scope)
    return GoogleDrive(gauth)

# =============================
# STREAMLIT UI
# =============================
def corrosion_ui():
    st.set_page_config(page_title="🚘 Corrosion Detector", layout="centered")
    st.title("🧠 Vehicle Corrosion Detection")

    uploaded_file = st.file_uploader("📷 Upload a Car Image", type=["jpg", "png", "jpeg"])
    confidence = st.slider("🎯 Confidence Threshold", value=CONFIDENCE_DEFAULT, step=0.05, min_value=0.05, max_value=1.0)

    if uploaded_file is not None:
        st.image(uploaded_file, caption="📷 Uploaded Image", use_container_width=True)

        if st.button("🚀 Run Detection"):
            with st.spinner("Downloading model & running corrosion detection..."):
                start_time = time.time()

                # 🔽 Download YOLOv8 model from Hugging Face
                model_dir = tempfile.mkdtemp()
                model_path = os.path.join(model_dir, "best_crk.pt")

                response = requests.get(MODEL_URL)
                if response.status_code == 200:
                    with open(model_path, "wb") as f:
                        f.write(response.content)
                else:
                    st.error("❌ Failed to download model from Hugging Face.")
                    st.stop()

                # 🔄 Save uploaded file to temp directory
                temp_dir = tempfile.mkdtemp()
                input_path = os.path.join(temp_dir, uploaded_file.name)
                with open(input_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                # 🧠 Load model
                model = YOLO(model_path)

                # 🔍 Run inference
                results = model.predict(input_path, conf=confidence)
                boxes = results[0].boxes

                # 🖼️ Draw results
                img = cv2.imread(input_path)
                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    conf = box.conf[0].item()
                    label = f"corrosion {conf:.2f}"
                    cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)
                    cv2.putText(img, label, (xyxy[0], xyxy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # # 💾 Save result locally to run_detection/
                # os.makedirs(LOCAL_SAVE_DIR, exist_ok=True)
                # result_filename = f"result_{uploaded_file.name}"
                # result_path = os.path.join(LOCAL_SAVE_DIR, result_filename)
                # cv2.imwrite(result_path, img)

                # 📷 Show result
                st.image(img, caption="🧪 Detection Result", use_container_width=True)

                # 🔐 Upload to Google Drive using folder ID
                drive = authenticate_drive(JSON_PATH)
                gfile = drive.CreateFile({'title': result_filename, 'parents': [{'id': DRIVE_FOLDER_ID}]})
                gfile.SetContentFile(result_path)
                gfile.Upload()

                elapsed = time.time() - start_time
                st.success("✅ Detection complete and uploaded to your Drive folder.")
                st.info(f"⏱️ Total Time: {elapsed:.2f} seconds")

                # ⬇️ Download Button
                with open(result_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Result Image",
                        data=f,
                        file_name=result_filename,
                        mime="image/jpeg"
                    )



