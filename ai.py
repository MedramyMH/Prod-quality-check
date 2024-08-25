import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import streamlit as st
import contextlib

# Suppress Keras prints
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = os.dup(1)
        os.dup2(devnull.fileno(), 1)
        try:
            yield
        finally:
            os.dup2(old_stdout, 1)

# Function to train a new CNN model
def train_cnn_model(data_dir):
    st.write("Training a new CNN model...")

    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    # Suppress the output from this section
    with suppress_stdout():
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(512, 512),
            batch_size=32,
            class_mode='binary',
            subset='training'
        )
        validation_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=(512, 512),
            batch_size=32,
            class_mode='binary',
            subset='validation'
        )

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Model training
    st.write("Starting model training...")
    model.fit(train_generator, epochs=10, validation_data=validation_generator)

    # Save the model
    model.save("product_quality_model.h5")
    st.write("Model training completed and saved.")
    return model

# Streamlit UI
st.title("Product Quality Checker")
st.write("Upload product images to check quality (OK/NOK)")

uploaded_ok_files = st.file_uploader("Upload OK Images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])
uploaded_nok_files = st.file_uploader("Upload NOK Images", accept_multiple_files=True, type=['png', 'jpg', 'jpeg'])

if st.button("Train New Model"):
    if uploaded_ok_files and uploaded_nok_files:
        # Assuming data directory setup here
        base_data_dir = "data_directory"
        model = train_cnn_model(base_data_dir)
    else:
        st.error("Please upload both OK and NOK images.")
