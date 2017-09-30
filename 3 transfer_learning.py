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
import numpy as np

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

# Initialize VGG16 using pre-trained weights on imagenet
model = VGG16(weights='imagenet', include_top=False,input_shape = (img_width, img_height,3))

# use transfer learning for re-training the last layers
# Freeze first 25 layers, so that we can retrain 26th and so on using our classes.
for layer in model.layers[:-5]:
   layer.trainable = False

# Adding our new layers 
top_layers = model.output
top_layers = Flatten(input_shape=model.output_shape[1:])(top_layers)
top_layers = Dense(num_of_classes, activation="relu",input_shape=(num_of_classes,))(top_layers)
top_layers = Dropout(0.5)(top_layers)
top_layers = Dense(num_of_classes, activation="relu",input_shape=(num_of_classes,))(top_layers)
top_layers = Dense(num_of_classes, activation="softmax")(top_layers)

# Add top layers on top of freezed (not re-trained) layers of VGG16
model_final = Model(input = model.input, output = top_layers)

# Compile the model
model_final.compile(loss = "binary_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])

# Initialize test and training data
train_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

test_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

train_generator = train_datagen.flow_from_directory(
train_data_dir,
target_size = (img_height, img_width),
batch_size = batch_size, 
class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(
validation_data_dir,
target_size = (img_height, img_width),
class_mode = "categorical")

# Save our model using specified conditions
checkpoint = ModelCheckpoint("vgg16_12.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

# Re-train our layers
model_final.fit_generator(
train_generator,
samples_per_epoch = nb_train_samples,
epochs = epochs,
validation_data = validation_generator,
nb_val_samples = nb_validation_samples,
callbacks = [checkpoint, early]
)