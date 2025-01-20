import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
from keras import utils, datasets, layers, models
import tensorflow_datasets as tfds
import matplotlib.image as mpimg
from tensorflow import data as tf_data


#Loading the data
image_data = tf.keras.utils.image_dataset_from_directory("/Users/rishvikkambhampati/Desktop/snake image classification/images")

#Partitioning the data into train, validation, and test datasets
train_size = int(len(image_data)*.6)
val_size = int(len(image_data)*.2)
test_size = int(len(image_data)*.2)

train_dataset = image_data.take(train_size)
validation_dataset = image_data.skip(train_size).take(val_size)
test_dataset = image_data.skip(train_size+val_size).take(test_size)

'''print(f"Number of training samples: {len(list(train_dataset.unbatch()))}")
print(f"Number of validation samples: {len(list(validation_dataset.unbatch()))}")
print(f"Number of test samples: {len(list(test_dataset.unbatch()))}") '''

'''Resizing all of the images so that they are the same shape and size with dimensions (150, 150) 
and batching the images into groups of 64. Shape should be (64, 150, 150, 3), which is 
(batch size, image_height, image_width, color channels) so 3 represents RGB color channel
'''
resize_fn = keras.layers.Resizing(150, 150)
train_dataset = train_dataset.map(lambda x, y: (resize_fn(x), y))
validation_dataset = validation_dataset.map(lambda x, y: (resize_fn(x), y))
test_dataset = test_dataset.map(lambda x, y: (resize_fn(x), y))

batch_size = 64
train_dataset = train_dataset.unbatch().batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()
validation_dataset = validation_dataset.unbatch().batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()
test_dataset = test_dataset.unbatch().batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()

#Visualize an image from the dataset
for batch in train_dataset.take(1):
    # Select the first image from the first batch and convert the image tensor values into a numpy array with pixel units from 0-255
    single_image = batch[0].numpy().astype("uint8")

# Remove the extra batch dimension as shape of single_image is (1, 150, 150, 3)
single_image = single_image[0]

# Display the image
'''plt.imshow(single_image)
plt.show()'''

#Models
#Model 1
model1 = models.Sequential([
    #These are the convolutional layers, creating the kernels necessary while pooling it to retain necessary information in as less spatial area as possible
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(150,150,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    #Flattening layers
    layers.Flatten(),

    #Dense layers
    layers.Dense(128, activation = "relu"),
    layers.Dropout(0.5),

    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),

    # Output layer
    layers.Dense(11, activation='softmax')
])

#Fitting the model to the training data
'''model1.compile(optimizer="adam", 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model1.fit(train_dataset,
                     epochs=20,
                     validation_data=validation_dataset) '''

#Model 2 (Data Augmentation)
model2 = models.Sequential([
    layers.RandomFlip("horizontal", input_shape=(150, 150, 3)),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomTranslation(0.2, 0.2),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(11, activation='softmax')
])

def augmented(train_ds, data_augmentation):
  # Display original and augmented images
  plt.figure(figsize=(10, 10))
  for image, _ in train_ds.take(1):
      original_image = image[0]
      ax = plt.subplot(2, 2, 1)
      plt.imshow(original_image / 255)
      plt.axis('off')
      ax.set_title('Original Image')

  # Display augmented images
  for i in range(2, 5):
      ax = plt.subplot(2, 2, i)
      augmented_image = data_augmentation(tf.expand_dims(original_image, 0))
      plt.imshow(augmented_image[0] / 255)
      plt.axis('off')
      ax.set_title(f'Augmented Image {i - 1}')

  plt.show()

data_augmentation2 = tf.keras.Sequential([
  tf.keras.layers.RandomRotation(0.2),
])
#augmented(train_dataset, data_augmentation2)


#Model 3
i = keras.Input(shape=(150, 150, 3))

scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
x = scale_layer(i)
preprocessor = keras.Model(inputs = i, outputs = x)

model3 = models.Sequential([
    preprocessor,

    # augmentation layers
    layers.RandomFlip("horizontal", input_shape=(150, 150, 3)),
    layers.RandomRotation(0.2),

    layers.Conv2D(64, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(512, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(512, activation=tf.keras.layers.LeakyReLU(alpha=0.1)),
    layers.Dropout(0.3),
    layers.Dense(11, activation='softmax')
])

IMG_SHAPE = (150, 150, 3)
base_model = keras.applications.MobileNetV3Large(input_shape=(150,150,3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

i = keras.Input(shape=IMG_SHAPE)
x = base_model(i, training = False)
base_model_layer = keras.Model(inputs = i, outputs = x)

model4 = models.Sequential([
    # augmentation layers
    layers.RandomFlip("horizontal", input_shape=(150, 150, 3)),
    layers.RandomRotation(0.2),

    base_model_layer,
    layers.GlobalMaxPooling2D(),
    layers.Dropout(0.2),
    layers.Dense(11, activation='softmax'),  # outputs the final classification
])

model2.compile(optimizer="adam", 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model2.fit(train_dataset,
                     epochs=20,
                     validation_data=validation_dataset) 

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Data Augmentation Model')
plt.legend()
plt.show()

test_loss, test_accuracy = model2.evaluate(test_dataset)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

