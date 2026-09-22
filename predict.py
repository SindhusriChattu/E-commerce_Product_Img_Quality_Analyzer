import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

MODEL_PATH = "product_image_quality_cnn.keras"

model = tf.keras.models.load_model(MODEL_PATH)


def predict_image(image_path):

    img = image.load_img(
        image_path,
        target_size=(224, 224)
    )

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(img_array)[0][0]

    if prediction >= 0.5:
        result = "POOR QUALITY"
        confidence = prediction
    else:
        result = "GOOD QUALITY"
        confidence = 1 - prediction

    return result, confidence


result, confidence = predict_image("test.jpg")

print("Prediction:", result)
print("Confidence:", round(confidence * 100, 2), "%")
