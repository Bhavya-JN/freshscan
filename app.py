import json
import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

# Load model
model = load_model("food_freshness_model.keras")

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

st.title("🍎 Food Freshness Detection")

st.write("Upload a fruit image to check whether it is Fresh or Rotten.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose a fruit image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Resize image
    img = image.resize((224, 224))

    # Convert to array
    img_array = np.array(img)

    # Normalize
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    # Find predicted class
    predicted_index = np.argmax(prediction[0])
    predicted_class = class_names[predicted_index]

    confidence = float(prediction[0][predicted_index]) * 100

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")