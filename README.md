# Image Classifier for Sorting Food and Non-Food Images 

## Overview
![Photosort Visual Display For Github Repo](https://github.com/jjudyyang/ML_photosort/assets/25236683/79539833-e639-4e5c-8842-dc30374f8879)

This project aims to automatically classify images into two categories: "food" and "not food.". It uses a Convolutional Neural Network (CNN) implemented with TensorFlow to perform image classification. The objective is to sort photos from your camera roll or any other source into these two categories, making it easier to manage image collection. After classifying the images, it also sorts the "food" images by date and location.

## Prerequisites

Before running the project, ensure you have the following prerequisites:

- Python 3.x
- The required Python libraries (NumPy, Pillow, TensorFlow, and scikit-learn) can be installed via pip:
    ```
    pip install numpy pillow tensorflow scikit-learn
    ```

## Usage

### 1. Organize Your Images

- Create a folder on your local machine (e.g., "camera_roll") and place the images you want to classify into this folder.

- Inside your project directory, create two subfolders: "food" and "not_food." The sorted images will be placed in these folders based on their classification.

### 2. Run the Image Classification Script

- Open a terminal or command prompt and navigate to your project directory.

- Run the Python script `classify_images.py` to start the image classification process:
    ```
    python classify_images.py
    ```

- You can customize the classification logic inside the script to improve accuracy.

### 3. Sort "Food" Images

- After classifying the images, the script will sort the "food" images by date and location into separate folders.

- The sorted "food" images will be available in subfolders within the "food" directory based on their date and location.

### 4. View the Results

- After the script finishes running, you will find the classified images in the "food" and "not_food" folders within your project directory, and sorted "food" images by date and location in subfolders within the "food" directory.

- The script will print the classification results for each image as "food" or "not food."

## Customizing the Model

- You can fine-tune the CNN model architecture and hyperparameters within the `classify_images.py` script to improve classification accuracy.

- Adjust the folder structure, path names, and naming conventions to meet your specific requirements.

## Acknowledgments
This project was created as a demonstration of image classification using a CNN and to make my much loved hobby of writing Google Reviews a bit faster (and in other aspects slower) 
