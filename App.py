import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import cv2
import tempfile
import joblib

# Load the ResNet model
model = load_model('/home/saher/Projects/Egyptian Hieroglyphs Classification/ResNet_model.h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the label encoder
le = joblib.load('/home/saher/Projects/Egyptian Hieroglyphs Classification/label_encoder.joblib')

# Title
st.title('Egyptian Hieroglyphs Classification! 📜🖌️')

# Header
st.header('Upload an image of an Egyptian hieroglyph to classify it and detect the hieroglyphs.')

# Function to preprocess and predict the image
def predict_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))  # Resize to model input shape
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize
    prediction = model.predict(image)
    return prediction

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
        
    # Display the uploaded image in Streamlit, with a set width
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', width=300)
        
    # Make prediction
    prediction = predict_image(temp_file_path)
    predicted_class = np.argmax(prediction)
        
    # Display the prediction result
    st.write(f"Prediction: {le.inverse_transform([predicted_class])[0]}")
else:
    st.write('Please upload an appropriate image to predict.')