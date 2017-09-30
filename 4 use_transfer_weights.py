from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model 
from keras import applications
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

k.set_image_dim_ordering('tf')

img_width = 64
img_height = 64
train_data_dir = "dataset/training_set"
validation_data_dir = "dataset/test_set"
nb_train_samples = 600
nb_validation_samples = 100
batch_size = 100
epochs = 3
num_of_classes = 2

if __name__ == "__main__":

    ## RECREATE THE MODEL THE WEIGHTS HAVE BEEN TRAINED ON!
    model = VGG16(weights=None, include_top=False,input_shape = (img_width, img_height,3))

    for layer in model.layers[:-5]:
        layer.trainable = False
    top_layers = model.output
    top_layers = Flatten(input_shape=model.output_shape[1:])(top_layers)
    top_layers = Dense(num_of_classes, activation="relu",input_shape=(num_of_classes,))(top_layers)
    top_layers = Dropout(0.5)(top_layers)
    top_layers = Dense(num_of_classes, activation="relu",input_shape=(num_of_classes,))(top_layers)
    top_layers = Dense(num_of_classes, activation="softmax")(top_layers)
    model_final = Model(input = model.input, output = top_layers)
    # load weights
    model_final.load_weights('vgg16_12.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model_final.compile(optimizer=sgd, loss='categorical_crossentropy')

    # Test image
    test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model_final.predict(test_image)
    if result[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'
    print(prediction)