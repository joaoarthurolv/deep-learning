from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow.keras.utils as tf_utils

classificador = Sequential()
classificador.add(Conv2D(64, (3,3), input_shape = (128, 128, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Conv2D(128, (3,3), input_shape = (128, 128, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Flatten())

classificador.add(Dense(units = 128, activation= 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation= 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 4, activation= 'softmax'))

classificador.compile(optimizer= 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

gerador_treinamento = ImageDataGenerator(rescale = 1./255, rotation_range = 7,
                                         horizontal_flip = True,
                                         shear_range = 0.2,
                                         height_shift_range = 0.07,
                                         zoom_range = 0.2)

gerador_teste = ImageDataGenerator(rescale = 1./255)

base_treinamento = gerador_treinamento.flow_from_directory('dataset/AugmentedAlzheimerDataset',
                                                           target_size = (128,128),
                                                           batch_size = 64,
                                                           class_mode = 'categorical')

base_teste = gerador_teste.flow_from_directory('dataset/OriginalDataset',
                                               target_size=(128,128),
                                               batch_size = 64,
                                               class_mode = 'categorical')

classificador.fit_generator(base_treinamento, steps_per_epoch = 33984 / 64, epochs = 100, 
                            validation_data = base_teste, validation_steps = 6400 / 64) 

