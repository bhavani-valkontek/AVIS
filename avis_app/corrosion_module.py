# import streamlit as st
# from ultralytics import YOLO
# import cv2
# import numpy as np
# import tempfile
# import json
# from PIL import Image
# import os
# import requests

# # =============================
# # CONFIGURATION
# # =============================
# MODEL_URL = "https://huggingface.co/babbilibhavani/scartch_detection/resolve/main/best_crk.pt"
# # CONFIDENCE_DEFAULT = 0.3

# # =============================
# # Download model from Hugging Face
# # =============================
# def download_model_from_huggingface(url, save_path):
#     response = requests.get(url)
#     if response.status_code == 200:
#         with open(save_path, "wb") as f:
#             f.write(response.content)
#         return save_path
#     else:
#         raise Exception(f"Failed to download model from Hugging Face. Status code: {response.status_code}")

# # =============================
# # Run YOLO Inference
# # =============================
# def run_inference(image_path, model_path): #conf_threshold
#     model = YOLO(model_path)
#     results = model.predict(source=image_path, imgsz=640, save=False) # conf=conf_threshold,
#     output = results[0]

#     image = cv2.imread(image_path)
#     image_draw = image.copy()
#     detection_data = []

#     if output.boxes is not None and len(output.boxes) > 0:
#         for box in output.boxes.data.tolist():
#             x1, y1, x2, y2, conf, cls = box
#             class_name = output.names[int(cls)]
#             cv2.rectangle(image_draw, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
#             label = f"{class_name}: {conf:.2f}"
#             cv2.putText(image_draw, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

#             detection_data.append({
#                 "class": class_name,
#                 # "confidence": float(f"{conf:.4f}"),
#                 "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
#             })
#     return image_draw, detection_data

# # =============================
# # Streamlit UI
# # =============================
# def corrosion_ui():
#     st.title("🚘 Vehicle Corrosion Detection")
#     st.markdown("Upload a vehicle image to detect corrosion using YOLOv8 model.")

#     # Download model
#     with st.spinner("📦 Downloading model from Hugging Face..."):
#         try:
#             model_path = download_model_from_huggingface(MODEL_URL, "best_crk.pt")
#         except Exception as e:
#             st.error(f"❌ Failed to download model: {e}")
#             st.stop()

#     image_file = st.file_uploader("📷 Upload an Image", type=["jpg", "jpeg", "png"],key="corrosion_image_uploader")
#     # conf_threshold = st.slider("🎯 Confidence Threshold", 0.05, 1.0, CONFIDENCE_DEFAULT, 0.05)

#     if image_file is not None:
#         input_image = Image.open(image_file).convert("RGB")
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
#             input_image.save(tmp_img.name)
#             image_path = tmp_img.name

#         with st.spinner("⏳ Running Corrosion Detection..."):
#             output_image, detections = run_inference(image_path, model_path) #, conf_threshold

#             st.image(output_image, caption="🧪 Detection Result", channels="BGR", use_container_width=True)

#             # Save output image for download
#             output_path = "corrosion_detection_output.jpg"
#             cv2.imwrite(output_path, output_image)

#             if detections:
#                 json_path = "corrosion_results.json"
#                 with open(json_path, "w") as f:
#                     json.dump(detections, f, indent=2)

#                 st.download_button("⬇️ Download Results (JSON)", data=json.dumps(detections, indent=2),
#                                    file_name="corrosion_results.json", mime="application/json")
#                 with open(output_path, "rb") as img_file:
#                     st.download_button("⬇️ Download Output Image", img_file.read(),
#                                        file_name="corrosion_detection_output.jpg", mime="image/jpeg")
#             else:
#                 st.warning("❌ No corrosion detected in the image.")



import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import json
from PIL import Image
from huggingface_hub import hf_hub_download

# ===================================
# 🔐 Load Corrosion Model (Cached)
# ===================================

@st.cache_resource
def load_corrosion_model():
    try:
        model_path = hf_hub_download(
            repo_id="babbilibhavani/scartch_detection",
            filename="best_crk.pt",
            token=st.secrets["Api_key"]["Apikey"]
        )
    except Exception as e:
        st.error(f"❌ Error downloading corrosion model: {e}")
        return None

    model = YOLO(model_path)
    return model


corrosion_model = load_corrosion_model()

if corrosion_model is None:
    st.stop()


# ===================================
# 🎯 Run Inference
# ===================================

def run_inference(image_path):
    results = corrosion_model.predict(source=image_path, imgsz=640, save=False)
    output = results[0]

    image = cv2.imread(image_path)
    image_draw = image.copy()
    detection_data = []

    if output.boxes is not None and len(output.boxes) > 0:
        for box in output.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box
            class_name = output.names[int(cls)]

            cv2.rectangle(
                image_draw,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 0, 255),
                2
            )

            label = f"{class_name}: {conf:.2f}"
            cv2.putText(
                image_draw,
                label,
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

            detection_data.append({
                "class": class_name,
                "confidence": float(conf),
                "bbox": [
                    round(x1, 1),
                    round(y1, 1),
                    round(x2, 1),
                    round(y2, 1)
                ]
            })

    return image_draw, detection_data


# ===================================
# 🚀 Streamlit UI
# ===================================

def corrosion_ui():
    st.title("🚘 Vehicle Corrosion Detection")
    st.markdown("Upload a vehicle image to detect corrosion.")

    image_file = st.file_uploader(
        "📷 Upload Corrosion Image",
        type=["jpg", "jpeg", "png"],
        key="corrosion_upload_tab"   # Unique key
    )

    if image_file is not None:

        input_image = Image.open(image_file).convert("RGB")

        with st.spinner("⏳ Running Corrosion Detection..."):

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
                input_image.save(tmp_img.name)
                image_path = tmp_img.name

            output_image, detections = run_inference(image_path)

        st.subheader("📊 Detection Results")
        st.image(output_image,
                 caption="Detection Result",
                 channels="BGR",
                 use_container_width=True)

        if detections:

            st.success(f"✅ {len(detections)} Corrosion Area(s) Detected")

            json_data = json.dumps(detections, indent=2)

            st.download_button(
                "⬇️ Download Results (JSON)",
                data=json_data,
                file_name="corrosion_results.json",
                mime="application/json"
            )

            _, buffer = cv2.imencode(".jpg", output_image)

            st.download_button(
                "⬇️ Download Output Image",
                data=buffer.tobytes(),
                file_name="corrosion_detection_output.jpg",
                mime="image/jpeg"
            )

        else:
            st.warning("❌ No corrosion detected.")

