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
# def run_inference(image_path, model_path): #, conf_threshold
#     model = YOLO(model_path)
#     results = model.predict(source=image_path,imgsz=640, save=False) # conf=conf_threshold, 
#     output = results[0]

#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError("Failed to load image. Check image path or format.")

#     image_draw = image.copy()
#     detection_data = []

#     if output.boxes is not None and len(output.boxes) > 0:
#         for box in output.boxes.data.tolist():
#             x1, y1, x2, y2, conf, cls = box
#             class_name = output.names[int(cls)]

#             # Draw bounding box
#             cv2.rectangle(image_draw, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
#             label = f"{class_name}: {conf:.2f}"
#             cv2.putText(image_draw, label, (int(x1), int(y1) - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

#             detection_data.append({
#                 "class": class_name,
#                 # "confidence": float(f"{conf:.4f}"),
#                 "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
#             })

#     return image_draw, detection_data


# # ============================
# # 🚀 Streamlit Web Application
# # ============================
# def glass_ui():
#     st.title("🪟 Glass Break Detection")
#     st.markdown("Upload an image to detect broken glass using YOLOv8.")

#     hf_model_url = "https://huggingface.co/babbilibhavani/scartch_detection/resolve/main/best_gb.pt"

#     with st.spinner("📦 Downloading model from Hugging Face..."):
#         try:
#             model_file = download_model_from_huggingface(hf_model_url, "best_gb.pt")
#         except Exception as e:
#             st.error(f"❌ Failed to download model: {e}")
#             st.stop()

#     image_file = st.file_uploader("🖼️ Upload Image", type=["jpg", "jpeg", "png"],key="glass_image_upload")
#     # conf_threshold = st.slider("🎯 Confidence Threshold", 0.05, 1.0, 0.25, 0.05,key="glas_conf_slider")

#     if image_file is not None:
#         final_image = Image.open(image_file).convert("RGB")

#         with st.spinner("⏳ Running Detection..."):
#             input_image_path = "uploaded_glass_image.jpg"
#             final_image.save(input_image_path)

#             try:
#                 output_image, detections = run_inference(input_image_path, model_file) #, conf_threshold
#             except Exception as e:
#                 st.error(f"❌ Error during inference: {e}")
#                 return

#             st.subheader("📊 Detection Results")
#             st.image(output_image, caption="🖼️ Detected Image", channels="BGR", use_container_width=True)

#             # Save output image
#             output_image_path = "glass_detection_output.jpg"
#             cv2.imwrite(output_image_path, output_image)

#             if detections:
#                 # Save JSON results
#                 json_path = "detection_results.json"
#                 with open(json_path, "w") as f:
#                     json.dump(detections, f, indent=2)

#                 st.success("✅ Detection completed successfully!")

#                 # JSON download button
#                 st.download_button(
#                     label="⬇️ Download Detection Results (JSON)",
#                     data=json.dumps(detections, indent=2),
#                     file_name="glass_detection_results.json",
#                     mime="application/json"
#                 )

#                 # Image download button
#                 with open(output_image_path, "rb") as img_file:
#                     st.download_button(
#                         label="⬇️ Download Detected Image (JPG)",
#                         data=img_file.read(),
#                         file_name="glass_detection_output.jpg",
#                         mime="image/jpeg"
#                     )
#             else:
#                 st.warning("⚠️ No broken glass detected.")


# # ============================
# # 🔁 Main Entry Point
# # # ============================
# # if __name__ == "__main__":
# #     glass_ui()


import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import json
from PIL import Image
from huggingface_hub import hf_hub_download

# ===================================
# 🔐 Load Glass Model (Cached)
# ===================================

@st.cache_resource
def load_glass_model():
    try:
        model_path = hf_hub_download(
            repo_id="babbilibhavani/scartch_detection",
            filename="best_gb.pt",
            token=st.secrets["Api_key"]["Apikey"]
        )
    except Exception as e:
        st.error(f"❌ Error downloading glass model: {e}")
        return None

    model = YOLO(model_path)
    return model


glass_model = load_glass_model()

if glass_model is None:
    st.stop()


# ===================================
# 🎯 Run Inference
# ===================================

def run_inference(image_path):
    results = glass_model.predict(source=image_path, imgsz=640, save=False)
    output = results[0]

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to load image.")

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

def glass_ui():
    st.title("🪟 Glass Break Detection")
    st.markdown("Upload an image to detect broken glass.")

    image_file = st.file_uploader(
        "🖼️ Upload Glass Image",
        type=["jpg", "jpeg", "png"],
        key="glass_upload_tab"   # Unique key
    )

    if image_file is not None:

        input_image = Image.open(image_file).convert("RGB")

        with st.spinner("⏳ Running Glass Detection..."):

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
                input_image.save(tmp_img.name)
                image_path = tmp_img.name

            try:
                output_image, detections = run_inference(image_path)
            except Exception as e:
                st.error(f"❌ Error during inference: {e}")
                return

        st.subheader("📊 Detection Results")
        st.image(
            output_image,
            caption="Detected Image",
            channels="BGR",
            use_container_width=True
        )

        if detections:

            st.success(f"✅ {len(detections)} Broken Glass Area(s) Detected")

            json_data = json.dumps(detections, indent=2)

            st.download_button(
                "⬇️ Download Detection Results (JSON)",
                data=json_data,
                file_name="glass_detection_results.json",
                mime="application/json"
            )

            _, buffer = cv2.imencode(".jpg", output_image)

            st.download_button(
                "⬇️ Download Detected Image (JPG)",
                data=buffer.tobytes(),
                file_name="glass_detection_output.jpg",
                mime="image/jpeg"
            )

        else:
            st.warning("⚠️ No broken glass detected.")

