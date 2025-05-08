import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes

# ---- 🎨 Add Background & Style ----
st.markdown("""
    <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1534854638093-bada1813ca19?auto=format&fit=crop&w=1470&q=80");
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }
        .block-container {
            backdrop-filter: blur(4px);
            background-color: rgba(0, 0, 0, 0.4);
            padding: 2rem;
            border-radius: 1rem;
        }
        h1, h2, h3, p, label {
            color: #f1f1f1;
        }
        .stFileUploader {
            background-color: #ffffff10;
            border: 1px solid #cccccc40;
            border-radius: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---- 📦 Load Model ----
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"]
img_preprocess = weights.transforms()

@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
    model.eval()
    return model

model = load_model()

# ---- 🎯 Prediction Function ----
def make_prediction(img):
    img_processed = img_preprocess(img)
    with torch.no_grad():
        prediction = model(img_processed.unsqueeze(0))[0]
    prediction["labels"] = [f"{categories[label]} ({score:.2f})"
                            for label, score in zip(prediction["labels"], prediction["scores"])]
    return prediction

# ---- 🖼️ Draw Bounding Boxes ----
def create_image_with_bboxes(img_np, prediction):
    img_tensor = torch.tensor(img_np, dtype=torch.uint8)
    colors = ["red" if "person" in label else "lime" for label in prediction["labels"]]
    img_with_bboxes = draw_bounding_boxes(
        img_tensor, boxes=prediction["boxes"], labels=prediction["labels"],
        colors=colors, width=2, font_size=18
    )
    img_with_bboxes_np = img_with_bboxes.permute(1, 2, 0).cpu().numpy()
    return img_with_bboxes_np

# ---- 🖥️ Streamlit App ----
st.title("🔍 Object Detection with Faster R-CNN")
st.markdown("Upload an image and detect objects with confidence scores using **Faster R-CNN ResNet50 FPN V2**.")

upload = st.file_uploader("📤 Upload an Image (JPG/PNG):", type=["png", "jpg", "jpeg"])

if upload:
    img = Image.open(upload).convert("RGB")

    st.image(img, caption="🖼️ Uploaded Image", use_column_width=True)

    with st.spinner("🔎 Detecting objects..."):
        prediction = make_prediction(img)
        img_with_bbox = create_image_with_bboxes(np.array(img).transpose(2, 0, 1), prediction)

    # ---- 📊 Show Prediction with Bounding Boxes ----
    st.subheader("📌 Detected Objects")
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img_with_bbox)
    plt.axis("off")
    st.pyplot(fig, use_container_width=True)

    # ---- 📋 Prediction Details ----
    with st.expander("🧾 Prediction Details"):
        pred_clean = {
            "labels": prediction["labels"],
            "scores": [f"{score:.2f}" for score in prediction["scores"].tolist()]
        }
        st.json(pred_clean)
