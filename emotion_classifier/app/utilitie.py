import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import glob
'''
This script prepares the data to be used in compare.py
'''

model = tf.keras.models.load_model('emotions.h5')


def predict_emotion(filename):

         

        
            #geting the class names and categories 
            class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
            print(class_names)
            img= tf.keras.utils.load_img('./data/'+filename, target_size = (48,48), color_mode='rgb')
            

            # image must be converted into a numpy array to be used in the network
            img = np.array(img)
            print(img.shape)
            img = img.reshape((1,48,48,3))
            y_pred = model.predict(img)
            y_pred_labels = np.argmax(y_pred, axis=1)
            label_pred = class_names[int(y_pred_labels)]
            
            return label_pred


if __name__ == "__main__":
  print(predict_emotion('PrivateTest_687498.jpg'))