import cv2
import numpy as np
import os
import pandas as pd

csv_file_path = './train.csv'
images_directory = './train_copy' #cropped train images
 
df = pd.read_csv(csv_file_path)
df = df[['File Name', 'Category']]
image_size = (180, 180)  # Desired image size

images = []  # List to store images
labels = []

for _, row in df.iterrows():
    image_path = os.path.join(images_directory, row['File Name'])
    label = row['Category']

    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.resize(img, (180, 180))  # Assuming you want to resize
        images.append(img)
        labels.append(label)
    else:
        print(f"Warning: Could not load image {image_path}")

# Convert the lists to NumPy arrays
images_np = np.array(images)
labels_np = np.array(labels)

print('Shape of the images array:', images_np.shape)
print('Shape of the labels array:', labels_np.shape)

import pickle 

# Pickle the images array
with open('images_np.pickle', 'wb') as f:
    pickle.dump(images_np, f)

# Pickle the labels array
with open('labels_np.pickle', 'wb') as f:
    pickle.dump(labels_np, f)