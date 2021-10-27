import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds

'''
This script prepares the data to be used in compare.py
'''

batch_size = 128
img_height = 48
img_width = 48

data_dir= '../data/train/'
            
#feeding the script with data in batches

train_ds = tf.keras.utils.image_dataset_from_directory(
data_dir,
validation_split=0.2,
subset="training",
seed=123,
image_size=(img_height, img_width),
batch_size=batch_size)



#seting a side validation data
val_ds = tf.keras.utils.image_dataset_from_directory(
data_dir,
validation_split=0.2,
subset="validation",
seed=123,
image_size=(img_height, img_width),
batch_size=batch_size)

#geting the class names and categories 
train_class_names = train_ds.class_names
print(train_class_names)

val_class_names = val_ds.class_names
print(val_class_names)

#normalizing the data
normalization_layer = tf.keras.layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))


num_classes = 7

#reading the test data
data_dir= '../data/test'
test_ad = tf.keras.utils.image_dataset_from_directory(
data_dir,
validation_split=0.2,
subset="validation",
seed=123,
image_size=(img_height, img_width),
batch_size=batch_size)   

train_class_names = train_ds.class_names
print(train_class_names)

val_class_names = val_ds.class_names
print(val_class_names)


