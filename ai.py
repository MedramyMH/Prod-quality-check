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
MODEL_DIR = "models"
UPLOADS_DIR = "uploads"

# Ensure necessary directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Custom callback to update training progress
class StreamlitProgressCallback(Callback):
    def __init__(self, progress_bar, status_text, start_time):
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.start_time = start_time

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            elapsed_time = time.time() - self.start_time
            remaining_epochs = self.params['epochs'] - (epoch + 1)
            time_per_epoch = elapsed_time / (epoch + 1)
            remaining_time = time_per_epoch * remaining_epochs

            self.progress_bar.progress((epoch + 1) / self.params['epochs'])
            self.status_text.write(f"Epoch {epoch + 1}/{self.params['epochs']} - "
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
    
    # Save the trained model
    model_path = os.path.join(MODEL_DIR, 'product_quality_model.h5')
    model.save(model_path)
    st.write("Model training completed and saved at:", model_path)
    return model

# Function to load an existing model
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
    model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.h5')]
    selected_model = st.selectbox("Select a model to load:", model_files)
    if selected_model:
        model_path = os.path.join(MODEL_DIR, selected_model)
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

# # Input for directory path
# base_data_dir = st.text_input("Enter the base directory path")
# ok_dir = os.path.join(base_data_dir, "OK")
# nok_dir = os.path.join(base_data_dir, "NOK")

# # Button to train a new model
# if st.button("Train New Model"):
#     if os.path.exists(ok_dir) and os.path.exists(nok_dir):
#         st.write(f"Using data from: {base_data_dir}")
#         model = train_cnn_model(base_data_dir)
#     else:
#         st.error("Please ensure the OK and NOK directories exist within the provided base directory path.")
# Input for directory path using text input
base_data_dir = st.text_input("Enter the base directory path")

# Check if directories are valid
ok_dir = os.path.join(base_data_dir, "OK")
nok_dir = os.path.join(base_data_dir, "NOK")

# Button to train a new model
if st.button("Train New Model"):
    if os.path.exists(ok_dir) and os.path.exists(nok_dir):
        st.write(f"Using data from: {base_data_dir}")
        model = train_cnn_model(base_data_dir)
    else:
        st.error("Please ensure the OK and NOK directories exist within the provided base directory path.")

# Uploading an existing model
model = select_model()

# Uploading a new image for prediction
uploaded_image = st.file_uploader("Upload a new product image:")
if uploaded_image and model:
    image_path = os.path.join(UPLOADS_DIR, uploaded_image.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())
    prediction = predict_quality(model, image_path)
    st.write(f"The quality of the product is: {prediction}")

# Option to use the camera for live prediction
use_camera = st.checkbox("Use Camera for Live Prediction")
if use_camera and model:
    st.write("Using camera to capture product image...")
    # Capture image using webcam
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
