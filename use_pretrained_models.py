import io
import os
from base64 import b64encode,b64decode

import seaborn as sb
import numpy as np
import keras
import re
import shutil
import requests

from flask import Flask
from flask import request

from keras.models import Sequential
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import Model
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet

print("Initialising models...")
resnet50 = ResNet50(weights='imagenet')
vgg16 = VGG16(weights='imagenet')
vgg19 = VGG19(weights='imagenet')
inceptionv3 = InceptionV3(weights='imagenet')
xception = Xception(weights='imagenet')
mobilenet = MobileNet(weights='imagenet')

img_path = "test.jpeg"

##base64_img = req_resize_server(img_path)
#img_data = io.BytesIO(b64decode(img_path))
img = image.load_img(img_path,target_size=(224,224))

array_img = image.img_to_array(img)
array_img = np.expand_dims(array_img, axis=0)
array_img = preprocess_input(array_img)

predictions = vgg16.predict(array_img)
decoded_predictions = decode_predictions(predictions)

print(str(decoded_predictions))