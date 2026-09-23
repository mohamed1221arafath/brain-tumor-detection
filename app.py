import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

IMG_SIZE = (150, 150)
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Load model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumor_model.keras")

model = load_model()

st.title("🧠 Brain Tumor MRI Classifier")
st.write("Upload a brain MRI scan and the model will predict the tumor type.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_container_width=True)

    # Preprocess
    img_resized = image.resize(IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    pred_index = np.argmax(predictions[0])
    predicted_class = class_names[pred_index]
    confidence = float(predictions[0][pred_index]) * 100

    st.subheader("Prediction Result")
    if predicted_class == "notumor":
        st.success(f"✅ No tumor detected ({confidence:.1f}% confidence)")
    else:
        st.warning(f"⚠️ Predicted: **{predicted_class.upper()}** ({confidence:.1f}% confidence)")

    # Show confidence for all classes
    st.subheader("Confidence breakdown")
    for i, class_name in enumerate(class_names):
        st.write(f"{class_name}: {predictions[0][i]*100:.1f}%")