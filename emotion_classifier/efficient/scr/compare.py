import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
from utilitie import train_ds
from utilitie import val_ds
from utilitie import test_ad
from utilitie import train_class_names
from utilitie import val_class_names
import matplotlib.pyplot as plt
#import base_line
from base_line import model
from classifier import model
#import classifier_test
from classifier_test import model
from sklearn.metrics import confusion_matrix

#load the base_line model
model = tf.keras.models.load_model('emotions.h5')
model.summary()

image_batch, label_batch = next(iter(val_ds))
y_pred = model.predict(image_batch)
print(image_batch.shape, label_batch.shape, y_pred.shape)
y_pred_labels = np.argmax(y_pred, axis=1)

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].numpy().astype("uint8"))
  label_pred = y_pred_labels[i]
  lable_true = label_batch[i]
  plt.title(val_class_names[label_pred]+ ' '+val_class_names[lable_true])
  plt.axis("off")
plt.show()


image_batch, label_batch = next(iter(test_ad))

y_pred_test = model.predict(image_batch)
y_pred_test_labels = np.argmax(y_pred, axis=1)

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].numpy().astype("uint8"))
  label_pred_test = y_pred_test_labels[i]
  lable_true = label_batch[i]
  plt.title(val_class_names[label_pred_test]+ ' '+val_class_names[lable_true])
  plt.axis("off")
plt.show()
plt.plot(model.history.history['loss'], label="traning loss")
plt.plot(model.history.history['val_loss'], label="validation loss")
plt.legend()
plt.show()
confusion_matrix(label_batch, y_pred)

#load model from classifier
model = tf.keras.models.load_model('emotions_v2.h5')
model.summary()

image_batch, label_batch = next(iter(val_ds))
y_pred = model.predict(image_batch)
print(image_batch.shape, label_batch.shape, y_pred.shape)
y_pred_labels = np.argmax(y_pred, axis=1)

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].numpy().astype("uint8"))
  label_pred = y_pred_labels[i]
  lable_true = label_batch[i]
  plt.title(val_class_names[label_pred]+ ' '+val_class_names[lable_true])
  plt.axis("off")
plt.show()

image_batch, label_batch = next(iter(test_ad))

y_pred_test = model.predict(image_batch)
y_pred_test_labels = np.argmax(y_pred, axis=1)

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].numpy().astype("uint8"))
  label_pred_test = y_pred_test_labels[i]
  lable_true = label_batch[i]
  plt.title(val_class_names[label_pred_test]+ ' '+val_class_names[lable_true])
  plt.axis("off")
plt.show()
plt.plot(model.history['loss'], label="traning loss")
plt.plot(model.history['val_loss'], label="validation loss")
plt.legend()
plt.show()




#load model from classifier
model = tf.keras.models.load_model('emotions_v3.h5')
model.summary()
image_batch, label_batch = next(iter(val_ds))
y_pred = model.predict(image_batch)
print(image_batch.shape, label_batch.shape, y_pred.shape)
y_pred_labels = np.argmax(y_pred, axis=1)

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].numpy().astype("uint8"))
  label_pred = y_pred_labels[i]
  lable_true = label_batch[i]
  plt.title(val_class_names[label_pred]+ ' '+val_class_names[lable_true])
  plt.axis("off")
plt.show()

image_batch, label_batch = next(iter(test_ad))

y_pred_test = model.predict(image_batch)
y_pred_test_labels = np.argmax(y_pred, axis=1)

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].numpy().astype("uint8"))
  label_pred_test = y_pred_test_labels[i]
  lable_true = label_batch[i]
  plt.title(val_class_names[label_pred_test]+ ' '+val_class_names[lable_true])
  plt.axis("off")
plt.show()
plt.plot(model.history['loss'], label="traning loss")
plt.plot(model.history['val_loss'], label="validation loss")
plt.legend()
plt.show()
