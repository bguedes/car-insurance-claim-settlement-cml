import base64
import json
import pickle as pk

import efficientnet.keras as efn
import keras
import numpy as np
import requests
from joblib import load
from keras.applications import ResNet50, imagenet_utils
from keras.applications.vgg16 import VGG16
from keras.utils.data_utils import get_file
from PIL import Image
from tensorflow.keras.utils import img_to_array, load_img

CLASS_INDEX = None

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

def predict(img_path, categ_list):
    image = load_img(img_path, target_size=(224, 224))
    img = prepare_image(image, target=(224, 224))
    model = VGG16(input_shape=(224, 224,3), weights="imagenet")
    out = model.predict(img)
    preds = get_predictions(out, top=5)
    for pred in preds[0]:
        if pred[0:2] in categ_list:
            print(pred[0:2])
            print("Successful. Proceeding to damage assessment...")
            return {"car_detected": "true"}
    print("The entered image is a not a car. Please try again. Consider a different angle or lighting.")
    return {"car_detected": "false"}

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

def detectCarImage(args):
  
    imageBase64Encoded = args["imageBase64"]
    base64_img_bytes = imageBase64Encoded.encode('utf-8')
    image = base64.decodebytes(base64_img_bytes)
  
    with open('car_model_cat_list.pk', 'rb') as f:
        categ_count = pk.load(f)

    categ_list = [k for k, v in categ_count.most_common()[:25]]

    return predict(image, categ_list)
