import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os


# Use an absolute path for the model to avoid issues when running the script from different directories.
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'waste_classifier_model.keras')
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
IMG_SIZE = 224 

# Model Loading 
# We use st.cache_resource to load the model only once, which speeds up the app on subsequent runs.
@st.cache_resource
# Loads the trained Keras model.
def load_model():
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

#  Preprocessing Function 
# Preprocesses the uploaded image to match the model's input requirements.
def preprocess_image(image):
   
    #  Resize the image
    image = image.resize((IMG_SIZE, IMG_SIZE))
    # Convert to NumPy array
    image_array = np.asarray(image)
    # Rescale pixel values from [0, 255] to [0, 1]
    image_array = image_array / 255.0
    # Add a batch dimension
    # The model expects a batch of images, so we add a dimension: (H, W, C) -> (1, H, W, C)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


# --- Streamlit App UI ---
st.title("Smart-Sort: Waste Classifier ♻️")
st.write(
    "Upload an image of a piece of trash, and the model will predict "
    "which category it belongs to: cardboard, glass, metal, paper, plastic, or general trash."
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # --- Display the Uploaded Image ---
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    st.write("")

    # --- Make Prediction ---
    if model is not None:
        st.write("Classifying...")
        
        # 1. Preprocess the image
        processed_image = preprocess_image(image)
        
        # 2. Get model's prediction
        prediction = model.predict(processed_image)
        
        # 3. Get the predicted class index and confidence
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = np.max(prediction) * 100
        
        # --- Display the Result ---
        st.success(f"Prediction: **{predicted_class_name}**")
        st.info(f"Confidence: **{confidence:.2f}%**")