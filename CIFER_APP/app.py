import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🖼️",
    layout="centered"
)

# Class Names
CLASS_NAMES = [
    'Airplane', 'Automobile', 'Bird', 'Cat', 'Deer',
    'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
]

# Load Trained Model (Dynamic File Search)
import os

@st.cache_resource
def load_cifar_model():
    model_filename = "final_cifar10_cnn_model.keras"
    
    # Check if model exists in current folder
    if os.path.exists(model_filename):
        return tf.keras.models.load_model(model_filename, compile=False)
    
    # Search in all subfolders (like CIFER_APP)
    for root, dirs, files in os.walk("."):
        if model_filename in files:
            full_path = os.path.join(root, model_filename)
            return tf.keras.models.load_model(full_path, compile=False)
            
    raise FileNotFoundError(f"Could not find {model_filename} anywhere in repo.")

try:
    model = load_cifar_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model: {e}")

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
st.sidebar.info("Secure 4 Tech Internship - Phase 2 Deployment")
st.sidebar.markdown("**Dataset:** CIFAR-10")
st.sidebar.markdown("**Model:** Optimized CNN V2")

# Main Title & Description
st.title("🖼️ Deep Learning Image Classifier")
st.write("Upload an image belonging to one of the 10 CIFAR classes to get real-time predictions.")

# Image Processing Function
def preprocess_image(image_data):
    size = (32, 32)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    
    # Ensure RGB channels
    if img_array.ndim == 2:  # Grayscale
        img_array = np.stack((img_array,)*3, axis=-1)
    elif img_array.shape[2] == 4:  # RGBA
        img_array = img_array[:, :, :3]
        
    normalized_image_array = (img_array.astype(np.float32) / 255.0)
    data = np.expand_dims(normalized_image_array, axis=0)
    return data

# File Uploader
file = st.file_uploader("Choose an image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

if file is None:
    st.text("Please upload an image file.")
else:
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("🔍 Predict Class"):
        if model_loaded:
            with st.spinner('Analyzing Image...'):
                processed_img = preprocess_image(image)
                predictions = model.predict(processed_img)
                score = tf.nn.softmax(predictions[0])
                
                predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
                confidence = np.max(predictions[0]) * 100
                
                st.success(f"**Prediction:** {predicted_class}")
                st.info(f"**Confidence Score:** {confidence:.2f}%")
                
                # Class Probabilities Bar Chart
                st.subheader("📊 Class Probabilities")
                prob_dict = {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(10)}
                st.bar_chart(prob_dict)
        else:
            st.error("Model is not loaded properly.")
