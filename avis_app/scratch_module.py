# import pandas as pd
# import streamlit as st
# import torch
# import torchvision
# from torchvision.transforms import functional as F
# import numpy as np
# import cv2
# from PIL import Image
# import io
# import os
# import requests
# import time
# import json
# from datetime import datetime
# from pytz import timezone

# @st.cache_resource
# def load_model():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model_path = "best_maskrcnn_model.pth"
#     hf_url = "https://huggingface.co/babbilibhavani/scartch_detection/resolve/main/best_maskrcnn_model_k.pth"

#     if not os.path.exists(model_path):
#         try:
#             hf_token = st.secrets["Api_key"]["Apikey"]
#             headers = {"Authorization": f"Bearer {hf_token}"}
#             response = requests.get(hf_url, headers=headers)
#             if response.status_code == 200:
#                 with open(model_path, 'wb') as f:
#                     f.write(response.content)
#             else:
#                 st.error(f"Failed to download model. Status code: {response.status_code}")
#                 return None, None
#         except Exception as e:
#             st.error(f"Error downloading model: {e}")
#             return None, None

#     model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=True)
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)

#     in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
#     model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, 256, 2)

#     state_dict = torch.load(model_path, map_location=device)
#     model.load_state_dict(state_dict)

#     model.to(device)
#     model.eval()
#     return model, device

# model, device = load_model()

# def predict_image(image, threshold=0.2):
#     image_tensor = F.to_tensor(image).to(device)
#     original = np.array(image)
#     with torch.no_grad():
#         prediction = model([image_tensor])[0]
#     boxes, masks, scores = prediction['boxes'], prediction['masks'], prediction['scores']
#     final_masks = []
#     valid_scores = []
#     for i in range(len(scores)):
#         if scores[i] > threshold:
#             mask = masks[i, 0].cpu().numpy()
#             final_masks.append(mask)
#             valid_scores.append(float(scores[i]))
#     return original, final_masks, valid_scores

# def create_mask_overlay(original_img, masks, scores):
#     h, w = original_img.shape[:2]
#     mask_overlay = np.zeros((h, w, 3), dtype=np.uint8)
#     total_pixels = 0
#     for mask in masks:
#         binary = (mask > 0.5).astype(np.uint8)
#         red = np.zeros_like(mask_overlay)
#         red[:, :, 2] = binary * 255
#         mask_overlay = cv2.addWeighted(mask_overlay, 1.0, red, 0.5, 0)
#         total_pixels += np.sum(binary)
#     overlayed = cv2.addWeighted(original_img, 1.0, mask_overlay, 0.6, 0)
#     all_y, all_x = np.where(np.sum(masks, axis=0) > 0.5)
#     if len(all_x) > 0 and len(all_y) > 0:
#         x1, y1, x2, y2 = int(np.min(all_x)), int(np.min(all_y)), int(np.max(all_x)), int(np.max(all_y))
#         cv2.rectangle(overlayed, (x1, y1), (x2, y2), (0, 255, 0), 3)

#     severity = (total_pixels / (h * w)) * 100
#     # confidence = np.mean(scores) * 100
#     font_scale = w / 1000
#     text = f"Severity: {severity:.1f}% "  #| Confidence: {confidence:.1f}% 
#     text1 = f"Mask Pixels: {total_pixels}"
#     cv2.putText(overlayed, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
#     cv2.putText(overlayed, text1, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)

#     return Image.fromarray(mask_overlay), Image.fromarray(overlayed), severity, total_pixels # confidence

# def scratch_ui():
#     st.title("🚗 Vehicle Scratch Detection")
#     uploaded_file = st.file_uploader("Upload vehicle image", type=["jpg", "jpeg", "png"],key="scratch_image_uploader")

#     if uploaded_file:
#         image_data = uploaded_file.read()
#         image = Image.open(io.BytesIO(image_data)).convert("RGB")
#         st.image(image, caption="Original Image", use_container_width=True)

#         if st.button("Run Detection"):
#             with st.spinner("Detecting Scratches..."):
#                 start_time = time.time()
#                 original_np, masks, scores = predict_image(image)
#                 end_time = time.time()
#                 detection_time = end_time - start_time

#             if masks:
#                 mask_img, overlay_img, severity, total_pixels  = create_mask_overlay(original_np, masks, scores) #confidence
#                 st.subheader("Results:")
#                 st.write(f"🕒 Detection Time: {detection_time:.2f} seconds")
#                 col1, col2 = st.columns(2)
#                 col1.image(mask_img, caption="Scratch Mask", use_container_width=True)
#                 col2.image(overlay_img, caption="Result Overlay", use_container_width=True)

#                 st.subheader("Scratch Details")
#                 st.write(f"Severity: {severity:.1f}%")
#                 st.write(f"Pixels: {total_pixels}px")
#                 # st.write(f"Confidence: {confidence:.1f}%")

#                 # Prepare download filenames
#                 ist = timezone('Asia/Kolkata')
#                 timestamp = datetime.now(ist).strftime('%Y%m%d_%H%M%S')

#                 # Convert images for download
#                 buf_original = io.BytesIO()
#                 image.save(buf_original, format="JPEG")
#                 buf_original.seek(0)

#                 buf_mask = io.BytesIO()
#                 mask_img.save(buf_mask, format="JPEG")
#                 buf_mask.seek(0)

#                 buf_overlay = io.BytesIO()
#                 overlay_img.save(buf_overlay, format="JPEG")
#                 buf_overlay.seek(0)

#                 # Create CSV data
#                 result_data = pd.DataFrame([{
#                     "filename": f"overlay_{timestamp}.jpg",
#                     "severity": severity,
#                     # "confidence": confidence,
#                     "pixels": total_pixels,
#                     "timestamp": datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S")
#                 }])
#                 csv_buf = io.StringIO()
#                 result_data.to_csv(csv_buf, index=False)

#                 st.download_button("⬇️ Download Original Image", data=buf_original, file_name=f"original_{timestamp}.jpg", mime="image/jpeg")
#                 st.download_button("⬇️ Download Mask Image", data=buf_mask, file_name=f"mask_{timestamp}.jpg", mime="image/jpeg")
#                 st.download_button("⬇️ Download Overlay Image", data=buf_overlay, file_name=f"overlay_{timestamp}.jpg", mime="image/jpeg")
#                 st.download_button("⬇️ Download Detection Data (CSV)", data=csv_buf.getvalue(), file_name="scratch_results.csv", mime="text/csv")

#                 st.success("✅ All results ready to download!")
#                 st.write("_____________________________________")
#                 st.markdown("""
#                 <span style='color: white;'>___________________@</span>
#                 <span style='color: orange; font-weight: bold;'>Valkontek Embedded Services</span>
#                 """, unsafe_allow_html=True)
#             else:
#                 st.warning("No scratches detected.")




import pandas as pd
import streamlit as st
import torch
import torchvision
from torchvision.transforms import functional as F
import numpy as np
import cv2
from PIL import Image
import io
import os
import time
from datetime import datetime
from pytz import timezone
from huggingface_hub import hf_hub_download

# ===============================
# MODEL LOADING
# ===============================

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model_path = hf_hub_download(
            repo_id="babbilibhavani/scartch_detection",
            filename="best_maskrcnn_model_k.pth",
            token=st.secrets["Api_key"]["Apikey"]
        )
    except Exception as e:
        st.error(f"Error downloading model: {e}")
        return None, None

    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, 256, 2
    )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model, device


model, device = load_model()

if model is None:
    st.stop()


# ===============================
# PREDICTION FUNCTION
# ===============================

def predict_image(image, threshold=0.2):
    image_tensor = F.to_tensor(image).to(device)
    original = np.array(image)

    with torch.no_grad():
        prediction = model([image_tensor])[0]

    boxes = prediction['boxes']
    masks = prediction['masks']
    scores = prediction['scores']

    final_masks = []
    valid_scores = []

    for i in range(len(scores)):
        if scores[i] > threshold:
            mask = masks[i, 0].cpu().numpy()
            final_masks.append(mask)
            valid_scores.append(float(scores[i]))

    return original, final_masks, valid_scores


# ===============================
# MASK OVERLAY
# ===============================

def create_mask_overlay(original_img, masks, scores):
    h, w = original_img.shape[:2]
    mask_overlay = np.zeros((h, w, 3), dtype=np.uint8)
    total_pixels = 0

    for mask in masks:
        binary = (mask > 0.5).astype(np.uint8)
        red = np.zeros_like(mask_overlay)
        red[:, :, 2] = binary * 255
        mask_overlay = cv2.addWeighted(mask_overlay, 1.0, red, 0.5, 0)
        total_pixels += np.sum(binary)

    overlayed = cv2.addWeighted(original_img, 1.0, mask_overlay, 0.6, 0)

    if len(masks) > 0:
        combined_mask = np.sum(masks, axis=0)
        all_y, all_x = np.where(combined_mask > 0.5)
        if len(all_x) > 0 and len(all_y) > 0:
            x1, y1 = int(np.min(all_x)), int(np.min(all_y))
            x2, y2 = int(np.max(all_x)), int(np.max(all_y))
            cv2.rectangle(overlayed, (x1, y1), (x2, y2), (0, 255, 0), 3)

    severity = (total_pixels / (h * w)) * 100

    font_scale = w / 1000
    cv2.putText(overlayed, f"Severity: {severity:.1f}%", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)

    return Image.fromarray(mask_overlay), Image.fromarray(overlayed), severity, total_pixels


# ===============================
# UI
# ===============================

def scratch_ui():
    st.title("🚗 Vehicle Scratch Detection")

    uploaded_file = st.file_uploader(
        "Upload vehicle image",
        type=["jpg", "jpeg", "png"],
        key="scratch_image_uploader"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Original Image", use_container_width=True)

        if st.button("Run Detection"):
            with st.spinner("Detecting Scratches..."):
                start_time = time.time()
                original_np, masks, scores = predict_image(image)
                detection_time = time.time() - start_time

            if masks:
                mask_img, overlay_img, severity, total_pixels = create_mask_overlay(
                    original_np, masks, scores
                )

                st.write(f"🕒 Detection Time: {detection_time:.2f} sec")

                col1, col2 = st.columns(2)
                col1.image(mask_img, caption="Scratch Mask", use_container_width=True)
                col2.image(overlay_img, caption="Result Overlay", use_container_width=True)

                st.write(f"Severity: {severity:.1f}%")
                st.write(f"Pixels: {total_pixels}px")

                st.success("Detection completed successfully")
            else:
                st.warning("No scratches detected.")


scratch_ui()

