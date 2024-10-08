import streamlit as st
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback
import numpy as np
import time
import cv2

# Constants
IMG_SIZE = (512, 512)
UPLOADS_DIR = "uploads"
BASE_DATA_DIR = "temp_training_data"

# Ensure necessary directories exist
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(BASE_DATA_DIR, exist_ok=True)

# Custom callback to update training progress
class StreamlitProgressCallback(Callback):
    def __init__(self, progress_bar, status_text, start_time):
        super().__init__()
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.start_time = start_time

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            elapsed_time = time.time() - self.start_time
            remaining_epochs = self.params['epochs'] - (epoch + 1)
            time_per_epoch = elapsed_time / (epoch + 1)
            remaining_time = time_per_epoch * remaining_epochs

            progress_value = (epoch + 1) / self.params['epochs']
            self.progress_bar.progress(progress_value)
            st.write(f"Epoch {epoch + 1}/{self.params['epochs']} - "
                     f"loss: {logs['loss']:.4f} - accuracy: {logs['accuracy']:.4f} - "
                     f"ETA: {remaining_time:.2f} seconds")

# Function to train a new CNN model
def train_cnn_model(data_dir):
    st.write("Training a new CNN model...")
    
    # Data generators for training and validation
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode='binary',
        subset='training'
    )
    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )
    
    # Model definition
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name='input_layer'),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
        
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Progress display setup
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()

    # Callback instance
    progress_callback = StreamlitProgressCallback(progress_bar, status_text, start_time)

    model.fit(train_generator, epochs=10, validation_data=validation_generator, callbacks=[progress_callback])
    
    # Save the trained model locally
    model_path = os.path.join(BASE_DATA_DIR, 'product_quality_model.h5')
    model.save(model_path)
    st.write("Model training completed and saved at:", model_path)
    return model

# Function to load an existing model from Google Drive in Colab
def load_existing_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        st.write("Loaded model from:", model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model. Error: {e}")
        return None

# Function to list and select a model from the models directory
def select_model():
    model_files = [f for f in os.listdir(BASE_DATA_DIR) if f.endswith('.h5')]
    selected_model = st.selectbox("Select a model to load:", model_files)
    if selected_model:
        model_path = os.path.join(BASE_DATA_DIR, selected_model)
        return load_existing_model(model_path)
    return None

# Function to make predictions
def predict_quality(model, image_path):
    image = load_img(image_path, target_size=IMG_SIZE)
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return "OK" if prediction[0][0] > 0.5 else "NOK"

# Streamlit UI
st.title("Product Quality Checker")
st.write("Upload product images to check quality (OK/NOK)")

uploaded_ok_files = st.file_uploader("Upload OK Images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
uploaded_nok_files = st.file_uploader("Upload NOK Images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

if st.button("Train New Model"):
    if uploaded_ok_files and uploaded_nok_files:
        ok_dir = os.path.join(BASE_DATA_DIR, "OK")
        nok_dir = os.path.join(BASE_DATA_DIR, "NOK")

        os.makedirs(ok_dir, exist_ok=True)
        os.makedirs(nok_dir, exist_ok=True)

        # Save uploaded OK images
        for file in uploaded_ok_files:
            with open(os.path.join(ok_dir, file.name), "wb") as f:
                f.write(file.getbuffer())

        # Save uploaded NOK images
        for file in uploaded_nok_files:
            with open(os.path.join(nok_dir, file.name), "wb") as f:
                f.write(file.getbuffer())

        st.write(f"Training with data from: {BASE_DATA_DIR}")
        with st.spinner("Training model..."):
            model = train_cnn_model(BASE_DATA_DIR)
    else:
        st.error("Please upload both OK and NOK images.")

# Load an existing model
model = select_model()

# Uploading a new image for prediction
uploaded_image = st.file_uploader("Upload a new product image:")
if uploaded_image and model:
    image_path = os.path.join(UPLOADS_DIR, uploaded_image.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())
    with st.spinner("Making prediction..."):
        prediction = predict_quality(model, image_path)
        st.write(f"The quality of the product is: {prediction}")

# Option to use the camera for live prediction
use_camera = st.checkbox("Use Camera for Live Prediction")
if use_camera and model:
    st.write("Using camera to capture product image...")
    with st.spinner("Capturing image..."):
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imwrite("camera_image.jpg", frame)
                cap.release()
                prediction = predict_quality(model, "camera_image.jpg")
                st.image(frame, caption=f"Prediction: {prediction}")
            else:
                st.error("Unable to capture image from camera.")
        else:
            st.error("Unable to access camera.")
