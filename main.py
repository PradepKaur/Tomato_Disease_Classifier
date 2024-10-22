import os
import json
import gdown
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Set up working directory and paths
working_dir = os.path.dirname(os.path.abspath(__file__))
model_file = 'tomatoes_disease_prediction_model.h5'
model_path = f"{working_dir}/{model_file}"
class_indices_path = f"{working_dir}/class_indices.json"

# Google Drive file ID for the model
file_id = "1-71_uDEa0URjFg7V4QmSP4vvVqRIOX4N"  # Replace with your actual file ID
url = f"https://drive.google.com/uc?id={file_id}"

# Check if the model already exists locally; if not, download it
if not os.path.exists(model_path):
    st.write("Downloading the model from Google Drive...")
    gdown.download(url, model_path, quiet=False)

# Load the pretrained model
model = tf.keras.models.load_model(model_path)

# Load the class indices
with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)


# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the Image
    img = Image.open(image_path)

    # Ensure the image is in RGB format (convert from RGBA if needed)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize the image
    img = img.resize(target_size)

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Scale the image values to [0,1]
    img_array = img_array.astype('float32') / 255.

    return img_array


# Function to Predict the Class of the Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# Streamlit App
st.title('🍅🌿 Tomato Disease Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')
