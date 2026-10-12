"""Explainable diabetic retinopathy research demo (not a diagnostic device)."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL_PATH = Path(__file__).resolve().parent / "models" / "efficientnet_finetuned.keras"
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
IMAGE_SIZE = (224, 224)
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120B")

st.set_page_config(page_title="Explainable DR Screening | Research Demo", page_icon="👁️", layout="wide")

@st.cache_resource(show_spinner="Loading EfficientNetB0 model...")
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return tf.keras.models.load_model(str(MODEL_PATH), compile=False)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Match Colab: RGB, brightness>10 bounding box, LANCZOS pad to 224."""
    image = image.convert("RGB")
    arr = np.asarray(image)
    brightness = arr.mean(axis=2)
    mask = brightness > 10
    if mask.any():
        ys, xs = np.where(mask)
        image = image.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    image = ImageOps.pad(image, IMAGE_SIZE, method=Image.Resampling.LANCZOS, color=(0, 0, 0))
    return np.asarray(image, dtype=np.uint8)


def predict(model, image: np.ndarray) -> np.ndarray:
    # Keras EfficientNetB0 includes input rescaling; do NOT divide by 255.
    batch = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)
    return np.asarray(model(batch, training=False)[0], dtype=np.float32)


def gradcam(model, image: np.ndarray, class_index: int) -> np.ndarray:
    """Reproduce the trained inference path through top_conv and the head."""
    backbone = next(
        layer for layer in model.layers
        if isinstance(layer, tf.keras.Model) and "efficientnet" in layer.name.lower()
    )
    extractor = tf.keras.Model(backbone.input, backbone.get_layer("top_conv").output)
    batch = tf.convert_to_tensor(image[None, ...], dtype=tf.float32)

    with tf.GradientTape() as tape:
        features = extractor(batch, training=False)
        tape.watch(features)
        x = backbone.get_layer("top_bn")(features, training=False)
        x = backbone.get_layer("top_activation")(x)
        x = model.get_layer("global_average_pooling2d")(x)
        x = model.get_layer("dropout")(x, training=False)
        scores = model.layers[-1](x)
        score = scores[:, class_index]

    grads = tape.gradient(score, features)
    if grads is None:
        raise RuntimeError("Grad-CAM gradients unavailable for this model.")
    channel_weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(features[0] * channel_weights, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = tf.image.resize(heatmap[..., None], IMAGE_SIZE).numpy().squeeze()
    return np.clip(heatmap, 0, 1)


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, opacity: float = 0.42) -> Image.Image:
    colored = (matplotlib.colormaps["jet"](heatmap)[..., :3] * 255).astype(np.float32)
    base = image.astype(np.float32)
    # Mask affects display only, NOT the model's actual prediction or attention.
    retinal_region = base.mean(axis=2) > 15
    alpha = (opacity * retinal_region)[..., None]
    result = (1 - alpha) * base + alpha * colored
    return Image.fromarray(np.uint8(np.clip(result, 0, 255)))


def generate_report(probabilities: np.ndarray, predicted_index: int, language: str) -> str:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Set it in your .env file.")

    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    values = "\n".join(f"{label}: {float(value) * 100:.2f}%" for label, value in zip(CLASS_NAMES, probabilities))
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=650,
        
messages = [
    {
        "role": "system",
        "content": (
            "You are a friendly diabetic eye-health education assistant "
            "for people in rural India. "
            "Your task is to explain a retinal screening prediction "
            "and its Grad-CAM visualization in simple, reassuring language. "

            "Do not discuss model architecture, accuracy, F1-score, "
            "training data, or technical performance metrics. "

            "Explain the predicted diabetic retinopathy severity "
            "in everyday language without presenting it as a confirmed diagnosis. "

            "Give practical, safe suggestions based on the predicted severity. "
            "For No DR, encourage routine diabetic eye examinations. "
            "For Mild DR, encourage arranging an eye specialist appointment. "
            "For Moderate DR, recommend timely specialist evaluation. "
            "For Severe or Proliferative DR, recommend prompt ophthalmologist "
            "assessment because these categories may threaten vision. "

            "For any sudden vision loss, new flashes, many new floaters, "
            "or a dark curtain over vision, advise urgent medical evaluation. "

            "Explain Grad-CAM carefully: red/yellow regions indicate areas "
            "that influenced the prediction more strongly, while blue regions "
            "indicate lower influence. These colors do NOT confirm damage, "
            "bleeding, lesions, or disease location. "
            "Do not claim to have visually examined the image. "

            "Do not invent retinal findings, recommend medicines, "
            "or suggest treatment without a clinician. "
            "Keep the report short, clear, supportive, and practical. "
            "Include a brief note that the result needs professional confirmation."
        )
    },
    {
        "role": "user",
        "content": (
            f"Write a patient-friendly eye screening report in {language}.\n\n"

            f"Predicted DR severity: {CLASS_NAMES[predicted_index]}\n\n"

            "Explain these sections:\n"
            "1. Screening Result - What this severity category generally means.\n"
            "2. What You Should Do - Practical next steps based on severity.\n"
            "3. Grad-CAM Heatmap Explanation - Explain red, yellow, and blue "
            "colors as prediction influence, not confirmed disease.\n"
            "4. Eye Health Tips - General diabetes eye-care suggestions.\n"
            "5. Important Note - This is a screening prediction, "
            "not a medical diagnosis; consult an eye specialist.\n\n"

            "Do not mention model accuracy, class probabilities, "
            "F1-score, dataset, or technical model details. "
            "Do not claim specific retinal abnormalities were detected."
        )
    }
],

    )
    return response.choices[0].message.content or "No report returned."


st.title("👁️ Explainable AI for Diabetic Retinopathy Screening")
st.caption("EfficientNetB0 · Grad-CAM · Groq GenAI | Hackathon research prototype")
st.warning(
    "⚠️ Not a medical device or diagnostic service. Test accuracy: 69.09%; "
    "Severe DR recall: 37.9%. Do not use predictions to make healthcare decisions."
)

with st.sidebar:
    st.header("Settings")
    language = st.selectbox("GenAI report language", ["Hinglish (simple Hindi + English)", "English"])
    opacity = st.slider("Heatmap opacity", 0.10, 0.80, 0.42, 0.05)
    st.divider()
    st.markdown("**Classes**")
    for idx, name in enumerate(CLASS_NAMES):
        st.write(f"{idx} — {name}")
    st.caption("Grad-CAM highlights influential regions, not confirmed retinal lesions.")

uploaded = st.file_uploader("Upload a retinal fundus image", type=["png", "jpg", "jpeg"])
if uploaded is None:
    st.info("Upload a fundus image to view a prediction and Grad-CAM visualization.")
    st.stop()

try:
    original = Image.open(uploaded)
    processed = preprocess_image(original)
except (UnidentifiedImageError, OSError, ValueError) as exc:
    st.error(f"Unable to read the uploaded image: {exc}")
    st.stop()

try:
    model = load_model()
    probabilities = predict(model, processed)
    predicted_index = int(np.argmax(probabilities))
    heatmap = gradcam(model, processed, predicted_index)
    overlay = overlay_heatmap(processed, heatmap, opacity)
except Exception as exc:
    st.error(f"Model inference failed: {exc}")
    st.stop()

st.subheader("Model results")
left, right = st.columns(2)
with left:
    st.image(processed, caption="Preprocessed retinal image (224 × 224)", use_container_width=True)
with right:
    st.image(overlay, caption="Grad-CAM overlay for predicted class", use_container_width=True)

st.metric("Predicted severity", CLASS_NAMES[predicted_index])
st.metric("Highest model score (uncalibrated)", f"{float(probabilities[predicted_index]):.1%}")
st.caption("This score is not a calibrated probability of disease or a clinical confidence estimate.")
st.subheader("Five-class model scores")
st.bar_chart({name: float(score) for name, score in zip(CLASS_NAMES, probabilities)}, horizontal=True)

st.subheader("GenAI explanation")
st.write("The report is generated from the classifier scores only; the image is not sent to Groq.")
if st.button("✨ Generate Hinglish / English report", type="primary"):
    with st.spinner("Generating report using Groq..."):
        try:
            report = generate_report(probabilities, predicted_index, language)
            st.session_state["report"] = report
            st.session_state["report_image"] = uploaded.name
        except Exception as exc:
            st.error(f"GenAI report unavailable: {exc}")

if st.session_state.get("report_image") == uploaded.name and st.session_state.get("report"):
    st.markdown(st.session_state["report"])
    st.download_button("Download report (.txt)", st.session_state["report"], file_name="dr_explanation_report.txt", mime="text/plain")

st.divider()
st.caption("Research demonstration only. Any retinal screening concerns require assessment by a qualified eye-care professional.")
