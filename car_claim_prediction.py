import json
import csv
import keras
import numpy as np
import io
import os
import sys
import requests
import urllib.request
import pickle as pk

from keras.applications import ResNet50
from keras.applications.vgg16 import VGG16
import efficientnet.keras as efn
#from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import img_to_array
from keras.applications import imagenet_utils
from keras.utils.data_utils import get_file
#from keras.preprocessing.image import load_img
from tensorflow.keras.utils import load_img
from PIL import Image
from io import BytesIO
from collections import Counter, defaultdict

CLASS_INDEX = None

def load_model():
    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    global model
    #model = ResNet101(input_shape=(224, 224,3), weights="imagenet")
    #model = efn.EfficientNetB0(input_shape = (224, 224, 3), weights = 'imagenet')
    #model = ResNet50(input_shape=(224, 224,3), weights="imagenet")
    model = VGG16(input_shape=(224, 224,3), weights="imagenet")


def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    # return the processed image
    return image

def get_predictions(preds, top=5):
    global CLASS_INDEX
    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_predictions` expects a batch of predictions (i.e. a 2D array of shape (samples, 1000)). Found array with shape: ' + str(preds.shape))
    if CLASS_INDEX is None:
        fpath = get_file('imagenet_class_index.json',
            'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json',
            cache_subdir='models')
        CLASS_INDEX = json.load(open(fpath))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results

def predict():
    # initialize the data dictionary that will be returned from the
    # view
    # read the image in PIL format
    data = {}
    #filename='IMG_5492.jpg'
    filename='00021.jpg'

    #response = requests.get(im["url"])
    #with open(filename, 'wb') as f:
    #        f.write(response.content)

    image = load_img(filename, target_size=(224, 224))
    image = prepare_image(image, target=(224, 224))

    # classify the input image and then initialize the list
    # of predictions to return to the client
    preds = model.predict(image)
    results = imagenet_utils.decode_predictions(preds)
    data["predictions"] = []

    # loop over the results and add them to the list of
    # returned predictions
    for (imagenetID, label, prob) in results[0]:
          r = {"label": label, "probability": float(prob)}
          data["predictions"].append(r)
    # return the data dictionary as a JSON response
    jdata = json.dumps(data)

    sys.stdout.write('result : \n')
    sys.stdout.write(jdata)

    return jdata

def get_car_categories():
    d = defaultdict(float)
    img_list = os.listdir('car_ims')
    for i, img_path in enumerate(img_list):
        sys.stdout.write('img_path : ' + img_path)
        image = load_img('car_ims/' + img_path, target_size=(224, 224))
        img = prepare_image(image, target=(224, 224))
        #img = prepare_image_224('car-damage-dataset/data1a/training/01-whole/'+img_path)
        model = ResNet50(input_shape=(224, 224,3), weights="imagenet")
        out = model.predict(img)
        preds = get_predictions(out,top=5)
        for pred in preds[0]:
            d[pred[0:2]]+=pred[2]
        if(i%50==0):
            print(i,'/',len(img_list),'complete')

    print('d -> ', d)
    return Counter(d)

def pipe1(img_path, categ_list):
    image = load_img(img_path, target_size=(224, 224))
    #urllib.request.urlretrieve(img_path, 'image.jpg')
    img = prepare_image(image, target=(224, 224))
    out = model.predict(img)
    preds = get_predictions(out, top=5)
    print("Ensuring entered picture is a car...")
    for pred in preds[0]:
        if pred[0:2] in categ_list:
            print(pred[0:2])
            print("Successful. Proceeding to damage assessment...")
            return "Successful. Proceeding to damage assessment..."
    print("The entered image is a not a car. Please try again. Consider a different angle or lighting.")
    return "The entered image is a not a car. Please try again. Consider a different angle or lighting."


load_model()
#categ_count = get_car_categories()

#with open('model_cat_list.pk', 'wb') as f:
#    pk.dump(categ_count, f, -1)

with open('model_cat_list.pk', 'rb') as f:
    categ_count = pk.load(f)

categ_list = [k for k, v in categ_count.most_common()[:25]]

pipe1('00021.jpg', categ_list)

predict()
