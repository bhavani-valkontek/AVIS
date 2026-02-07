# # dent_module.py

# import streamlit as st
# from ultralytics import YOLO
# import cv2
# import numpy as np
# import tempfile
# import json
# from PIL import Image
# import os
# import requests

# # ===============================
# # 🎯 Download model from Hugging Face
# # ===============================
# def download_model_from_huggingface(url, save_path):
#     response = requests.get(url)
#     if response.status_code == 200:
#         with open(save_path, "wb") as f:
#             f.write(response.content)
#         return save_path
#     else:
#         raise Exception(f"Failed to download model from Hugging Face. Status code: {response.status_code}")

# # ===============================
# # 🎯 Function: Run YOLO Inference
# # ===============================
# def run_inference(image_path, model_path):  #conf_threshold
#     model = YOLO(model_path)
#     results = model.predict(source=image_path, imgsz=640, save=False) #conf=conf_threshold, 
#     output = results[0]

#     image = cv2.imread(image_path)
#     image_draw = image.copy()
#     detection_data = []

#     if output.boxes is not None and len(output.boxes) > 0:
#         for box in output.boxes.data.tolist():
#             x1, y1, x2, y2, conf, cls = box
#             class_name = output.names[int(cls)]

#             # Draw bounding box
#             cv2.rectangle(image_draw, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#             label = f"{class_name}: {conf:.2f}"
#             cv2.putText(image_draw, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

#             detection_data.append({
#                 "class": class_name,
#                 # "confidence": float(f"{conf:.4f}"),
#                 "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
#             })
#     return image_draw, detection_data

# # ============================
# # 🚀 Streamlit Web Application
# # ============================
# def dent_ui():
#     st.title("🔍Vehicle Dent Detection")
#     st.markdown("Upload an image or use your camera to detect car dents.")

#     hf_model_url = "https://huggingface.co/babbilibhavani/scartch_detection/resolve/main/best_model.pt"

#     with st.spinner("📦 Downloading model from Hugging Face..."):
#         try:
#             model_file = download_model_from_huggingface(hf_model_url, "best_model.pt")
#         except Exception as e:
#             st.error(f"❌ Failed to download model: {e}")
#             st.stop()

#     image_file = st.file_uploader("🖼️ Upload Image", type=["jpg", "jpeg", "png"],key="dent_image_uploader")
#     # conf_threshold = st.slider("🎯 Confidence Threshold", 0.05, 1.0, 0.25, 0.05)

#     if image_file is not None:
#         final_image = Image.open(image_file).convert("RGB")

#         with st.spinner("⏳ Running Detection..."):
#             with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
#                 final_image.save(tmp_img.name)
#                 tmp_img_path = tmp_img.name

#             output_image, detections = run_inference(tmp_img_path, model_file) #conf_threshold
#             st.subheader("📊 Detection Results")

#             st.image(output_image, caption="🖼️ Detected Image", channels="BGR", use_container_width=True)

#             output_image_path = "dent_detection_output.jpg"
#             cv2.imwrite(output_image_path, output_image)

#             if detections:
#                 json_path = "detection_results.json"
#                 with open(json_path, "w") as f:
#                     json.dump(detections, f, indent=2)

#                 st.download_button("⬇️ Download Results (JSON)", data=json.dumps(detections, indent=2),
#                                    file_name="detection_results.json", mime="application/json")
#                 with open(output_image_path, "rb") as img_file:
#                     st.download_button("⬇️ Download Output Image", img_file.read(),
#                                        file_name="dent_detection_output.jpg", mime="image/jpeg")
#             else:
#                 st.warning("❌ No dents detected in the image.")




# dent_module.py

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import json
from PIL import Image
import torch
from huggingface_hub import hf_hub_download

# ===================================
# 🔐 Load YOLO Model (Cached)
# ===================================

@st.cache_resource
def load_dent_model():
    try:
        model_path = hf_hub_download(
            repo_id="babbilibhavani/scartch_detection",
            filename="best_model.pt",
            token=st.secrets["Api_key"]["Apikey"]
        )
    except Exception as e:
        st.error(f"❌ Error downloading dent model: {e}")
        return None

    model = YOLO(model_path)
    return model


dent_model = load_dent_model()

if dent_model is None:
    st.stop()


# ===================================
# 🎯 Run Inference
# ===================================

def run_inference(image_path):
    results = dent_model.predict(source=image_path, imgsz=640, save=False)
    output = results[0]

    image = cv2.imread(image_path)
    image_draw = image.copy()
    detection_data = []

    if output.boxes is not None and len(output.boxes) > 0:
        for box in output.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box
            class_name = output.names[int(cls)]

            cv2.rectangle(image_draw, (int(x1), int(y1)),
                          (int(x2), int(y2)), (0, 255, 0), 2)

            label = f"{class_name}: {conf:.2f}"
            cv2.putText(image_draw, label,
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

            detection_data.append({
                "class": class_name,
                "confidence": float(conf),
                "bbox": [round(x1, 1), round(y1, 1),
                         round(x2, 1), round(y2, 1)]
            })

    return image_draw, detection_data


# ===================================
# 🚀 Streamlit UI
# ===================================

def dent_ui():
    st.title("🔍 Vehicle Dent Detection")
    st.markdown("Upload an image to detect vehicle dents.")

    image_file = st.file_uploader(
        "🖼️ Upload Dent Image",
        type=["jpg", "jpeg", "png"],
        key="dent_upload_tab"   # Unique key
    )

    if image_file is not None:

        final_image = Image.open(image_file).convert("RGB")

        with st.spinner("⏳ Running Dent Detection..."):

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
                final_image.save(tmp_img.name)
                tmp_img_path = tmp_img.name

            output_image, detections = run_inference(tmp_img_path)

        st.subheader("📊 Detection Results")
        st.image(output_image, caption="Detected Image",
                 channels="BGR", use_container_width=True)

        if detections:

            st.success(f"✅ {len(detections)} Dent(s) Detected")

            json_data = json.dumps(detections, indent=2)

            st.download_button(
                "⬇️ Download Detection Results (JSON)",
                data=json_data,
                file_name="dent_detection_results.json",
                mime="application/json"
            )

            _, buffer = cv2.imencode(".jpg", output_image)

            st.download_button(
                "⬇️ Download Output Image",
                data=buffer.tobytes(),
                file_name="dent_detection_output.jpg",
                mime="image/jpeg"
            )

        else:
            st.warning("❌ No dents detected.")

