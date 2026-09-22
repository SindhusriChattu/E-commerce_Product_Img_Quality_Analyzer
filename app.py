import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Product Image Quality Analyzer",
    page_icon="🛍️"
)

@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model(
        "product_image_quality_cnn.keras",
        compile=False
    )

try:
    model = load_cnn_model()
    st.success("CNN model loaded successfully!")

except Exception as e:
    st.error("CNN model could not be loaded.")
    st.code(str(e))
    st.stop()

model = load_cnn_model()

st.title("🛍️ E-commerce Product Image Quality Analyzer")

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

    img = img.resize((224, 224))

    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0][0]

    if prediction >= 0.5:
        result = "POOR QUALITY"
        confidence = prediction
    else:
        result = "GOOD QUALITY"
        confidence = 1 - prediction

    st.subheader("Analysis Result")
    st.write(f"### {result}")
    st.write(f"Confidence: {confidence * 100:.2f}%")
