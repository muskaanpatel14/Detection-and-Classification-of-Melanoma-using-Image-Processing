# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 02:11:08 2020

@author: Muskaan Patel
"""


# -*- coding: utf-8 -*-

#Part 1- Building the CNN

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


#Part 2 - Initialising the CNN

classifier = Sequential()

#Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape=(64,64,3), activation='relu'))

#Step 2 - Max Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))


#Step 3 - Flattening
classifier.add(Flatten())

#Step 4 - Adding Hidden and Iutput Layer
classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dense(output_dim=1, activation='sigmoid'))

#Step 5 - Compiling the CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Part 3 - Fitting the CNN to Image
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('train',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')
test_set = test_datagen.flow_from_directory('test',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')
classifier.fit_generator(training_set,
                    steps_per_epoch=2056,
                    epochs=25,
                    validation_data=test_set,
                    validation_steps=580)

classifier.save("classifier.h5")

from keras.preprocessing import image
import numpy as np

test_image = image.load_img('malignant.jpg',target_size=(64,64))
test_image=image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)
result = classifier.predict(test_image)

print(result)