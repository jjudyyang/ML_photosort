from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
import numpy as np
import cv2
import os
import shutil
import matplotlib.pyplot as plt
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

# Function to extract EXIF data
def extract_exif(image_path):
    exif_data = {}
    image = Image.open(image_path)
    info = image._getexif()
    if info:
        for tag, value in info.items():
            tag_name = TAGS.get(tag, tag)
            exif_data[tag_name] = value
    return exif_data

# Function to get GPS data from EXIF
def extract_gps_info(exif_data):
    gps_info = {}
    if "GPSInfo" in exif_data:
        for tag, value in exif_data["GPSInfo"].items():
            tag_name = GPSTAGS.get(tag, tag)
            gps_info[tag_name] = value
    return gps_info

# Define the directories for training and validation data
train_dir = '/Users/judy/Desktop/training'
validation_dir = '/Users/judy/Desktop/validation'
output_food_dir = 'output/food'

# Data augmentation and normalization
train_datagen = ImageDataGenerator(rescale=1/255)
validation_datagen = ImageDataGenerator(rescale=1/255)

# Load and prepare the training and validation datasets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(200, 200),
    batch_size=3,
    class_mode='binary'  # Food and Not Food
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(200, 200),
    batch_size=3,
    class_mode='binary'  # Food and Not Food
)

# Define the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    loss="binary_crossentropy",
    optimizer=RMSprop(learning_rate=0.001),
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=3,
    epochs=10,
    validation_data=validation_generator
)

# Process and sort images in the "food" folder based on EXIF data
if os.path.exists(output_food_dir):
    for root, dirs, files in os.walk(output_food_dir):
        for filename in files:
            img_path = os.path.join(root, filename)
            exif_data = extract_exif(img_path)
            gps_info = extract_gps_info(exif_data)

            if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info and "DateTimeOriginal" in exif_data:
                latitude = gps_info["GPSLatitude"]
                longitude = gps_info["GPSLongitude"]
                date = exif_data["DateTimeOriginal"]
                
                location_folder = f"{latitude[0]}_{latitude[1]}_{longitude[0]}_{longitude[1]}"
                date_folder = date.split()[0]
                
                location_date_folder = os.path.join(output_food_dir, location_folder, date_folder)
                os.makedirs(location_date_folder, exist_ok=True)
                
                shutil.move(img_path, os.path.join(location_date_folder, filename))
