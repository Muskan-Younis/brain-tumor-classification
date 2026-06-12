import cv2
import streamlit as st
import tensorflow as tf
from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np
from PIL import Image
import os

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Brain Tumor Classifier", layout="wide")

st.title("Brain Tumor Classification (MobileNetV2)")

IMG_SIZE = 224

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]
# -----------------------------
# GRAD-CAM FUNCTION
# -----------------------------
def generate_gradcam(model, img_array, last_conv_layer_name="Conv_1"):

    # Get MobileNetV2 base model
    base_model = model.layers[0]

    # Get last convolution layer
    last_conv_layer = base_model.get_layer(last_conv_layer_name)

    # Create grad model
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            last_conv_layer.output,
            model.layers[-1].output
        ]
    )

    # Forward pass
    with tf.GradientTape() as tape:

        # Pass image through base model first
        conv_outputs = base_model(img_array)

        tape.watch(conv_outputs)

        # Continue remaining layers manually
        x = conv_outputs

        for layer in model.layers[1:]:
            x = layer(x)

        predictions = x

        predicted_class = tf.argmax(predictions[0])

        loss = predictions[:, predicted_class]

    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Remove batch dimension
    conv_outputs = conv_outputs[0]

    # Weight channels
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    # Apply ReLU
    heatmap = tf.maximum(heatmap, 0)

    # Normalize
    max_val = tf.reduce_max(heatmap)

    if max_val > 0:
        heatmap /= max_val

    return heatmap.numpy()

# -----------------------------
# LIME FUNCTION
# -----------------------------
def generate_lime_explanation(model, img_array):

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        image=img_array[0].astype("double"),
        classifier_fn=model.predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=8,
        hide_rest=False
    )

    lime_image_result = mark_boundaries(
        temp / 255.0,
        mask
    )

    return lime_image_result
# -----------------------------
# IMAGE PREDICTION
# -----------------------------
st.subheader("MRI Image Analysis")

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:

    # -----------------------------
    # LOAD IMAGE
    # -----------------------------
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    img = np.array(image)

    # -----------------------------
    # PREPROCESSING
    # -----------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    enhanced = clahe.apply(gray)

    enhanced_rgb = cv2.cvtColor(
        enhanced,
        cv2.COLOR_GRAY2RGB
    )

    img_array = enhanced_rgb / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -----------------------------
    # LOAD MODEL
    # -----------------------------
    if "model" in st.session_state:
        model = st.session_state["model"]

    elif os.path.exists("brain_tumor_model.h5"):
        model = tf.keras.models.load_model(
            "brain_tumor_model.h5"
        )

    else:
        st.warning("⚠️ Train the model first")
        st.stop()

    # -----------------------------
    # PREDICTION
    # -----------------------------
    prediction = model.predict(img_array)

    class_names = CLASS_NAMES

    predicted_class = class_names[
        np.argmax(prediction)
    ]

    confidence = np.max(prediction)

    # -----------------------------
    # GRAD-CAM
    # -----------------------------
    heatmap = generate_gradcam(model, img_array)

    heatmap = cv2.resize(
        heatmap,
        (img.shape[1], img.shape[0])
    )

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    superimposed_img = heatmap * 0.4 + img
    # -----------------------------
    # LIME EXPLANATION
    # -----------------------------
    lime_result = generate_lime_explanation(
        model,
        img_array
    )

    # -----------------------------
    # TOP METRICS
    # -----------------------------
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Predicted Tumor Type",
            value=predicted_class.upper()
        )

    with col2:
        st.metric(
            label="Confidence",
            value=f"{confidence*100:.2f}%"
        )

    # Confidence warnings
    if confidence >= 0.85:
        st.success("High confidence prediction")

    elif confidence >= 0.60:
        st.warning("Moderate confidence prediction")

    else:
        st.error("Low confidence prediction")

    st.markdown("---")

    # -----------------------------
    # EXPLANATION VISUALS
    # -----------------------------
    st.subheader("Explainable AI Visualizations")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(
            image,
            caption="Original MRI",
            use_container_width=True
        )

    with col2:
        st.image(
            superimposed_img.astype("uint8"),
            caption="Grad-CAM",
            use_container_width=True
        )

    with col3:
        st.image(
            lime_result,
            caption="LIME Explanation",
            use_container_width=True
        )
    # -----------------------------
    # CLASS PROBABILITIES
    # -----------------------------
    st.subheader("Prediction Probabilities")

    for i, class_name in enumerate(class_names):

        prob = prediction[0][i]

        st.write(
            f"{class_name}: {prob*100:.2f}%"
        )

        st.progress(float(prob))

    # -----------------------------
    # OPTIONAL PREPROCESSING VIEW
    # -----------------------------
    with st.expander("View CLAHE Enhanced MRI"):

        st.image(
            enhanced_rgb,
            caption="CLAHE Enhanced MRI",
            use_container_width=True
        )