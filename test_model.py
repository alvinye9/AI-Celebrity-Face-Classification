import cv2
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
import shutil
import pathlib

batch_size = 16
img_height = 180
img_width = 180

skip_preprocessing = True

if not(skip_preprocessing):
    # =================== CROP TEST IMAGES ================================
    # Path to the directory containing test images
    test_dir = './test'
    # Copy dataset so that future symbolic links wont modify original dataset
    copy_images_dir = './test_copy'

    # Create a copy of the original directory
    shutil.copytree(test_dir, copy_images_dir)
    # Update the original_images_dir to point to the copy
    test_dir = copy_images_dir

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Path to the input directory containing images
    input_directory = './test_copy'

    # Function to crop faces from an image and overwrite the original image
    def crop_faces_and_replace(image_path):
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
            return  # Skip images without faces
        
        # Find the largest face
        largest_face = max(faces, key=lambda x: x[2] * x[3])  # Compare based on area (width * height)
            
        # Crop and replace the detected faces in the original image
        x, y, w, h = largest_face
        cropped_face = img[y:y+h, x:x+w]
        cv2.imwrite(image_path, cropped_face)

    # Iterate through the images in the input directory
    for root, dirs, files in os.walk(input_directory):
        for file in files:
            image_path = os.path.join(root, file)
            # Crop faces and replace them in the original images
            crop_faces_and_replace(image_path)

    test_dir = pathlib.Path(test_dir).with_suffix('') 
    image_count = len(list(test_dir.glob('*/*.jpg')))
    print("Number of Test Images After Cropping: ", image_count)


# =================== LOAD AND TEST MODEL ================================
# Load the category CSV file
category_df = pd.read_csv('./category.csv')

# Create a dictionary mapping numerical labels to category names
class_names = dict(zip(category_df['#'], category_df['Category']))
print(class_names)

# Load the trained model
model = load_model('my_model.keras')

test_dir = './test_copy'


# Function to predict the class label for an image
def predict_image(image_path, model):
    img = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    category = class_names[np.argmax(score)]
    #print("{} most likely belongs to {} with a {:.2f} percent confidence.".format(image_path, category, 100 * np.max(score)))
    return category



# Create a list to store the results
results = []

# Iterate through the images in the test directory
for filename in os.listdir(test_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust based on supported file types
        image_path = os.path.join(test_dir, filename)
        # Predict the class label for the image
        predicted_class = predict_image(image_path, model)
        # Append the results to the list
        results.append({'id': filename, 'Category': predicted_class})

# Create a DataFrame from the list of results
results_df = pd.DataFrame(results)

# Remove the '.jpg' extension from the 'id' column
results_df['id'] = results_df['id'].str.replace('.jpg', '')

# # Replace numerical category labels with their corresponding names
# results_df['Category'] = results_df['Category'].map(class_names)

#converts id from numerical string to int
results_df['id'] = results_df['id'].astype(int)

# Sort by 'id' column
results_df = results_df.sort_values(by='id')  

#write results to csv file
results_df.to_csv('results.csv', index=False)

