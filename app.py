import threading

import pandas as pd
import streamlit as st
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from ml.dataset import CLASS_LABELS, CLASS_NAMES
from ml.gradcam import explain_image
from ml.train import make_model

st.set_page_config(
    page_title="Skin Lesion Research",
    page_icon="🔬",
    layout="wide",
)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 25_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


@st.cache_resource
def load_model():
    torch.set_num_threads(2)

    checkpoint = torch.load(
        "artifacts/fold_0_best.pt",
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint.get("class_names") != CLASS_NAMES:
        raise ValueError("Checkpoint class order does not match the app.")

    model = make_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, threading.Lock()


st.title("🔬 Skin Lesion Image Analysis")
st.caption("EfficientNet-B0 · HAM10000 · Grad-CAM")

st.warning(
    "Research use only. This is not a medical diagnosis or a medical "
    "device. Model scores may be uncalibrated, and heatmaps do not "
    "prove a prediction is correct."
)

uploaded = st.file_uploader(
    "Upload a dermatoscopic JPEG or PNG image",
    type=["jpg", "jpeg", "png"],
)

if uploaded is not None:
    if uploaded.size > MAX_UPLOAD_BYTES:
        st.error("The image must be no larger than 10 MB.")
        st.stop()

    try:
        with Image.open(uploaded) as source:
            if source.format not in {"JPEG", "PNG"}:
                st.error("Please upload a valid JPEG or PNG.")
                st.stop()

            if source.width * source.height > MAX_PIXELS:
                st.error("The image dimensions are too large.")
                st.stop()

            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()

    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        st.error("Could not read this image safely.")
        st.stop()

    st.image(image, caption="Uploaded image", width=350)

    if st.button("Analyze image", type="primary"):
        try:
            with st.spinner("Running inference and generating Grad-CAM..."):
                model, lock = load_model()

                # Grad-CAM hooks and gradients are shared mutable state.
                with lock:
                    index, probabilities, overlay_png = explain_image(
                        model,
                        image,
                        torch.device("cpu"),
                    )

            code = CLASS_NAMES[index]

            st.subheader("Model output")
            st.write(
                f"**Highest-scoring class:** {CLASS_LABELS[code]} "
                f"({code.upper()})"
            )
            st.write(f"**Model score:** {probabilities[index] * 100:.2f}%")

            left, right = st.columns(2)
            with left:
                st.image(image, caption="Uploaded image", use_container_width=True)
            with right:
                st.image(
                    overlay_png,
                    caption="Grad-CAM overlay",
                    use_container_width=True,
                )

            st.caption(
                "The heatmap uses the model's 224 × 224 input geometry. "
                "It is not a lesion boundary."
            )

            results = pd.DataFrame(
                {
                    "Class": CLASS_NAMES,
                    "Category": [CLASS_LABELS[c] for c in CLASS_NAMES],
                    "Score (%)": [round(p * 100, 2) for p in probabilities],
                }
            )

            st.subheader("Seven-class score breakdown")
            st.bar_chart(results.set_index("Class")["Score (%)"])
            st.dataframe(results, hide_index=True, use_container_width=True)

        except Exception as error:
            st.error("The model could not process this image.")
            st.exception(error)
