import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os
import pickle as pk

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.models import Model
from keras.layers import Flatten, Dense
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

from tensorflow import keras

def load_and_prep_image(filename, img_shape=224):

  # Read in target file (an image)
  img = tf.io.read_file(filename)

  # Decode the read file into a tensor & ensure 3 colour channels
  img = tf.image.decode_image(img, channels=3)

  # Resize the image (to the same size our model was trained on
  img = tf.image.resize(img, size = [img_shape, img_shape])

  # Rescale the image (get all values between 0 and 1)
  img = img/255.
  return img

def pred_and_plot(model, filename):

  # Import the target image and preprocess it
  img = load_and_prep_image(filename)

  # Make a prediction
  pred = model.predict(tf.expand_dims(img, axis=0))
  pred=pred.round()
  if pred==0:
    print("Car is damages. Proceeding to damage assessment...")
  else:
      print("Car is not damages. Please take another photo...")

#with open('car damage or not detection model.h5', 'rb') as f:
#    model = pk.load(f)

model = keras.models.load_model('Car Damage Model_1/car_damage_or_not_detection_model.h5')

pred_and_plot(model, "IMG_5536.jpg")

pred_and_plot(model, "IMG_5492.jpg")
