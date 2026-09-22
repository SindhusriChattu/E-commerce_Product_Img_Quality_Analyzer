import streamlit as st
import numpy as np
import tensorflow as tf

from PIL import Image


# -----------------------------
# Load Model
# -----------------------------

model = tf.keras.models.load_model(
    "product_image_quality_cnn.keras",
    compile=False
)


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Product Image Quality Analyzer",
    page_icon="🛍️",
    layout="centered"
)


# -----------------------------
# Title
# -----------------------------

st.title("🛍️ E-commerce Product Image Quality Analyzer")

st.write(
    "Upload a product image to check whether "
    "it meets the expected image-quality standard."
)


# -----------------------------
# Upload Image
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload Product Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(
        img,
        caption="Uploaded Product Image",
        use_container_width=True
    )

    # Resize
    resized_img = img.resize((224, 224))

    # Convert to array
    img_array = np.array(resized_img)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Prediction
    prediction = model.predict(img_array)[0][0]

    # Classification
    if prediction >= 0.5:

        result = "POOR QUALITY"
        confidence = prediction

    else:

        result = "GOOD QUALITY"
        confidence = 1 - prediction


    # -----------------------------
    # Result
    # -----------------------------

    st.subheader("Analysis Result")

    st.write(
        f"### {result}"
    )

    st.write(
        f"Confidence: {confidence * 100:.2f}%"
    )


    # -----------------------------
    # Recommendation
    # -----------------------------

    if result == "GOOD QUALITY":

        st.success(
            "The image appears suitable for a product listing."
        )

    else:

        st.warning(
            "The image may require improvement before "
            "being used for a product listing."
        )
