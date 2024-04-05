batch_size = 10
epochs = 6

images_path = './images_np.pickle'
labels_path = './labels_np.pickle'

import pickle
with open(images_path, 'rb') as f:
    images_np = pickle.load(f)
with open(labels_path, 'rb') as f:
    labels_np = pickle.load(f)

print('Shape of the images array:', images_np.shape)
print('Shape of the labels array:', labels_np.shape)

import tensorflow as tf
from keras.utils import to_categorical
from keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.applications.resnet50 import preprocess_input

import numpy as np

num_classes = 100
seed = 123

label_encoder = LabelEncoder()
labels_int = label_encoder.fit_transform(labels_np)

print("Number of Classes: ", len(np.unique(labels_int)))
# Convert labels to one-hot encoding
labels_one_hot = tf.keras.utils.to_categorical(labels_int, num_classes=num_classes)


# Create a TensorFlow dataset from the images and labels
dataset = tf.data.Dataset.from_tensor_slices((images_np, labels_one_hot))



# Normalization function
def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255., label

# Apply the normalization function to the dataset
dataset = dataset.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

# Example of splitting the dataset into 80% training and 20% validation
train_size = int(0.8 * len(images_np))
#train_dataset = dataset.take(train_size)
#val_dataset = dataset.skip(train_size)

# Split the dataset into training and validation sets
train_dataset = dataset.take(train_size).batch(batch_size)
val_dataset = dataset.skip(train_size).batch(batch_size)


# for image_batch, label_batch in dataset.take(1):
#     print(image_batch.shape, label_batch.shape)
    
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Resizing

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(180,180, 3)))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
  train_dataset, 
  validation_data=val_dataset,
  batch_size=batch_size, 
  epochs=epochs)

model.summary()


model_path = 'my_model.keras'

model.save(model_path)

print("Model saved successfully :)") 

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)


import matplotlib.pyplot as plt

plt.figure(3, figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig("loss.jpg")

plt.show()
