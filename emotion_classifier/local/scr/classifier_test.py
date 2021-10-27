import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd

#defining the patch size and properties of images 

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

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2D(64, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2D(64, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Conv2D(128, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),

  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(1024, activation='relu'),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(num_classes),
  ])



#calculating the loss function
model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

#fitting the model 
results_CT = model.fit(train_ds,
epochs=25,
batch_size=100,
validation_data=val_ds)

df = pd.DataFrame(results_CT.history)
df.to_csv('results_CT_model.csv')

#saving model for later use
model.save('emotions_v3.h5')

#loading the save model
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
plt.savefig('prediction_val_CT')

data_dir= '../data/test'
test_ad = tf.keras.utils.image_dataset_from_directory(
data_dir,
validation_split=0.2,
subset="validation",
seed=123,
image_size=(img_height, img_width),
batch_size=batch_size)   


image_batch, label_batch = next(iter(test_ad))

y_pred_test = model.predict(image_batch)
y_pred_test_labels = np.argmax(y_pred_test, axis=1)

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].numpy().astype("uint8"))
  label_pred_test = y_pred_test_labels[i]
  lable_true = label_batch[i]
  plt.title(val_class_names[label_pred_test]+ ' '+val_class_names[lable_true])
  plt.axis("off")
plt.show()
plt.savefig('prediction_test_CT')

plt.plot(results_CT.history['loss'], label="traning loss")
plt.plot(results_CT.history['val_loss'], label="validation loss")
plt.show()
plt.savefig('learning_cure_CT')