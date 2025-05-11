import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
import base64
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes

# -------------------- ✅ Background + Button --------------------
def set_background(image_file):
    with open(image_file, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    st.markdown(f"""
        <style>
            .stApp {{
                background-image: url("data:image/png;base64,{data}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
                overflow-x: hidden;
            }}
            .return-form {{
                position: fixed;
                top: 20px;
                right: 30px;
                z-index: 999999;
            }}
            .return-button {{
                background-color: #222;
                color: #f1f1f1;
                padding: 12px 24px;
                border: 2px solid #00b894;
                border-radius: 50px;
                font-size: 15px;
                font-weight: 600;
                transition: all 0.3s ease;
                cursor: pointer;
            }}
            .return-button:hover {{
                background-color: #00b894;
                color: #000;
                transform: scale(1.05);
            }}
        </style>
        <form class="return-form" method="get" action="https://www.insightbyakshay.in">
            <button class="return-button" type="submit">← Return to Portfolio</button>
        </form>
    """, unsafe_allow_html=True)

set_background("bg2.png")

# -------------------- 📦 Load Model --------------------
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"]
img_preprocess = weights.transforms()

@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
    model.eval()
    return model

model = load_model()

# -------------------- 🔍 Prediction --------------------
def make_prediction(img):
    img_processed = img_preprocess(img)
    with torch.no_grad():
        prediction = model(img_processed.unsqueeze(0))[0]
    prediction["labels"] = [f"{categories[label]} ({score:.2f})"
                            for label, score in zip(prediction["labels"], prediction["scores"])]
    return prediction

# -------------------- 📦 Draw Bounding Boxes --------------------
def create_image_with_bboxes(img_np, prediction):
    img_tensor = torch.tensor(img_np, dtype=torch.uint8)
    colors = ["red" if "person" in label else "lime" for label in prediction["labels"]]
    img_with_bboxes = draw_bounding_boxes(
        img_tensor, boxes=prediction["boxes"], labels=prediction["labels"],
        colors=colors, width=2, font_size=18
    )
    return img_with_bboxes.permute(1, 2, 0).cpu().numpy()

# -------------------- 🎯 UI Layout --------------------
st.markdown("<h1 style='color:white;'>🔍 Object Detection with Faster R-CNN</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:white;'>Upload an image to detect objects using <strong>Faster R-CNN ResNet50</strong> with confidence scores and bounding boxes.</p>", unsafe_allow_html=True)

upload = st.file_uploader("📤 Upload an Image (JPG/PNG):", type=["jpg", "jpeg", "png"])

if upload:
    img = Image.open(upload).convert("RGB")
    st.image(img, caption="📷 Uploaded Image", use_container_width=True)

    with st.spinner("🔎 Detecting objects..."):
        prediction = make_prediction(img)
        img_with_bbox = create_image_with_bboxes(np.array(img).transpose(2, 0, 1), prediction)

    st.subheader("📌 Detected Objects")
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(img_with_bbox)
    plt.axis("off")
    st.pyplot(fig, use_container_width=True)

    with st.expander("🧾 Prediction Details"):
        st.json({
            "labels": prediction["labels"],
            "scores": [f"{score:.2f}" for score in prediction["scores"].tolist()]
        })
