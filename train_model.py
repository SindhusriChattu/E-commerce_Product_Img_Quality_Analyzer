import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# Configuration
# -----------------------------

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

DATASET_PATH = "dataset"

# -----------------------------
# Data Preprocessing
# -----------------------------

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

validation_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

print("Class mapping:", train_data.class_indices)

# -----------------------------
# CNN Model
# -----------------------------

model = models.Sequential([
    
    layers.Input(shape=(224, 224, 3)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(1, activation="sigmoid")
])

# -----------------------------
# Compile
# -----------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# Training
# -----------------------------

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS
)

# -----------------------------
# Save Model
# -----------------------------

model.save("product_image_quality_cnn.keras")

print("Model saved successfully.")

# -----------------------------
# Evaluation
# -----------------------------

loss, accuracy = model.evaluate(validation_data)

print("Validation Accuracy:", accuracy)

# Predictions
predictions = model.predict(validation_data)

predicted_classes = (predictions > 0.5).astype(int).flatten()

true_classes = validation_data.classes

print("\nClassification Report:")
print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=list(validation_data.class_indices.keys())
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        true_classes,
        predicted_classes
    )
)

# -----------------------------
# Accuracy Graph
# -----------------------------

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Training vs Validation Accuracy")
plt.legend()

plt.savefig("accuracy_graph.png")
plt.show()

# -----------------------------
# Loss Graph
# -----------------------------

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Training vs Validation Loss")
plt.legend()

plt.savefig("loss_graph.png")
plt.show()
