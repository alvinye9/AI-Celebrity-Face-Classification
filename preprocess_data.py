import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import cv2

from tensorflow import keras
from keras import layers
from keras.models import load_model
import pathlib
import pandas as pd
import imghdr
from pathlib import Path
import shutil

# https://github.com/tensorflow/docs/blob/master/site/en/tutorials/images/classification.ipynb
#change between train and train_small
# ====================== CREATE NEW DIRECTORY WITH CLASS SUBDIRECTORIES ====================== 
train_dir = './train'
train_csv = 'train.csv'
copy_images_dir = './train_copy' #final dataset one used for training (after preprocessing)
new_root_dir = './structured_train' #dataset in subdirectory format (not used in final submission)
batch_size = 16
img_height = 180
img_width = 180
seed = 123 #for shuffling dataset
epochs = 15


# Load the CSV file
df = pd.read_csv(train_csv)
df = df[['File Name', 'Category']]

# The root directory where your current images are stored
original_images_dir = train_dir

# Copy dataset so that future symbolic links wont modify original dataset
# Create a copy of the original directory
shutil.copytree(original_images_dir, copy_images_dir)
# Update the original_images_dir to point to the copy
original_images_dir = copy_images_dir

# Create new root directory with class-based folder structure compatible with tensorflow
os.makedirs(new_root_dir, exist_ok=True)

for index, row in df.iterrows():
    # Directory for the current image's class
    class_dir = os.path.join(new_root_dir, row['Category'])
    os.makedirs(class_dir, exist_ok=True)
    
    # Original and new file paths
    original_file_path = os.path.join(original_images_dir, row['File Name'])
    new_file_path = os.path.join(class_dir, row['File Name'])
    
    # Create a symbolic link if the file doesn't already exist
    if not os.path.exists(new_file_path):
        os.symlink(os.path.abspath(original_file_path), new_file_path)

new_root_dir = pathlib.Path(new_root_dir).with_suffix('') #this directory has subdirectories representing the classes
image_count = len(list(new_root_dir.glob('*/*.jpg')))
print("Number of Images Before Filtering out Corrupted Ones: ", image_count)

# ======================  FILTER OUT IMAGES NOT RECOGNIZED BY TENSORFLOW ====================== 
image_extensions = [".jpg"]  # add there all your images file extensions
img_type_accepted_by_tf = ["bmp", "gif", "jpeg", "png"]
for subdir in Path(new_root_dir).iterdir():
    if subdir.is_dir():
        for filepath in subdir.rglob("*"):
            if filepath.suffix.lower() in image_extensions:
                img_type = imghdr.what(filepath)
                if img_type is None:
                    # print(f"{filepath} is not an image")
                    os.remove(filepath)  # Delete the invalid image file
                elif img_type not in img_type_accepted_by_tf:
                    # print(f"{filepath} is a {img_type}, not accepted by TensorFlow")
                    os.remove(filepath)  # Delete the invalid image file

image_count = len(list(new_root_dir.glob('*/*.jpg')))
print("Number of Images After Filtering out Corrupted Ones: ", image_count)

# ====================== CROP FACES AND DELETE IMAGES WITH NO FACE ====================== 
# https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
# Load the cascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Path to the input directory containing images
input_directory = new_root_dir

# Function to crop faces from an image and overwrite the original image
def crop_face_and_replace(image_path):
    # print(f"Processing image: {image_path}")
    # Read the input image
    img = cv2.imread(image_path)
    if img is None:
        # print(f"Error: Unable to read image from path: {image_path}")
        return  # Skip corrupted images
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Check if faces are detected
    if len(faces) == 0:
        # print(f"No faces detected in image: {image_path}")
        os.remove(image_path) #delete images with no face
        return  
    
    # Find the largest face
    largest_face = max(faces, key=lambda x: x[2] * x[3])  # Compare based on area (width * height)
    
    # Crop and replace the detected face in the original image
    x, y, w, h = largest_face
    cropped_face = img[y:y+h, x:x+w]
    cv2.imwrite(image_path, cropped_face)


# Iterate through the images in the input directory
for root, dirs, files in os.walk(input_directory):
    for file in files:
        image_path = os.path.join(root, file)
        # Crop faces and replace them in the original images
        crop_face_and_replace(image_path)

image_count = len(list(new_root_dir.glob('*/*.jpg')))
print("Number of Images After Cropping: ", image_count)
