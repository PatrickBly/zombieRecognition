from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
import tensorflow as tf
import os
import numpy as np
from pathlib import Path
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy
from keras import backend as K


#get images and directories
print(os.listdir('C:/Users/blystone_patrick/Downloads/Coding/Coding/electiveProgramming/zombieRecognition'))

data_dir = Path("C:/Users/blystone_patrick/Downloads/Coding/Coding/electiveProgramming/zombieRecognition")
train_dir = data_dir/'train'
validation_dir = data_dir/'validation'
test_dir = data_dir/'test'
epoch = 10

#format datasets
train_dataset = tf.keras.utils.image_dataset_from_directory(
    directory = 'C:/Users/blystone_patrick/Downloads/Coding/Coding/electiveProgramming/zombieRecognition/classNames',
    subset = "training",
    seed = 100,
    validation_split = 5/6,
    color_mode = 'rgb',
    labels = 'inferred',
    batch_size = 3,
    image_size = (175, 175), 
    label_mode = 'binary')
    
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    directory = 'C:/Users/blystone_patrick/Downloads/Coding/Coding/electiveProgramming/zombieRecognition/classNames',
    validation_split = 5/6,
    subset= "validation",
    seed = 100,
    image_size = (175, 175),
    batch_size = 3)

class_names = train_dataset.class_names
print(class_names)

for image_batch, labels_batch in train_dataset:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

#create model
model = Sequential()
model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
optimizer='rmsprop',
metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale = 1 / 175,
)

test_gen = ImageDataGenerator(
    rescale = 1 / 175
)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size = (175, 175),
    batch_size = 3,
    class_mode = 'binary'
)

validation_gen = test_gen.flow_from_directory(
    validation_dir,
    target_size = (175, 175),
    batch_size = 3,
    class_mode='binary'
)

model.fit_generator(
    train_datagen,
    steps_per_epoch = 6 // 3,
    epochs = 10,
    validation_data = validation_gen,
    validation_steps = 6 // 3
)

model.save_weights('zombieRecognition')

model = load_model('zombieRecognition')

model_img = load_img('', target_size=(175,175))
model_pic = np.array(model)
model_pic = model_pic / 175

#ValueError: Failed to find data adapter that can handle input: <class 'keras.preprocessing.image.ImageDataGenerator'>, <class 'NoneType'>