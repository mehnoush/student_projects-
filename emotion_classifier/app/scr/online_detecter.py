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

from sklearn.metrics import confusion_matrix

def predict():

        model = tf.keras.models.load_model('emotions_v3.h5')
        model.summary()

        data_dir= '../data/upload/'

        


        batch_size = 10
        img_height = 48
        img_width = 48
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

        return y_pred_test_labels