from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow.keras.utils as tf_utils

classificador = Sequential()
classificador.add(Conv2D(64, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Conv2D(128, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Conv2D(256, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Conv2D(512, (3,3), input_shape = (64, 64, 3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Flatten())

classificador.add(Dense(units = 128, activation= 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation= 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 4, activation= 'softmax'))

classificador.compile(optimizer= 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

gerador_treinamento = ImageDataGenerator(rescale = 1./255, # Normalização
                                         rotation_range = 7, 
                                         horizontal_flip = True,
                                         shear_range = 0.2, # Mudança de pixels pra outra direção
                                         height_shift_range = 0.07,
                                         zoom_range = 0.2)

gerador_teste = ImageDataGenerator(rescale = 1./255)

base_treinamento = gerador_treinamento.flow_from_directory('dataset/AugmentedAlzheimerDataset',
                                                           target_size = (64,64),
                                                           batch_size = 32,
                                                           class_mode = 'categorical')

base_teste = gerador_teste.flow_from_directory('dataset/OriginalDataset',
                                               target_size=(64,64),
                                               batch_size = 32,
                                               class_mode = 'categorical')

classificador.fit_generator(base_treinamento, steps_per_epoch = 10000 / 32, epochs = 100, 
                            validation_data = base_teste, validation_steps = 6400 / 32) 

base_treinamento.class_indices

# Teste Mild Demented
img_teste_mild_demented = tf_utils.load_img(
    'dataset/AugmentedAlzheimerDataset/MildDemented/5d23012d-5531-4a38-90a6-8400b913615d.jpg', target_size = (64,64))

img_teste_mild_demented = tf_utils.img_to_array(img_teste_mild_demented)
img_teste_mild_demented /= 255
img_teste_mild_demented = np.expand_dims(img_teste_mild_demented, axis = 0)

# Teste Very Mild Demented
img_teste_very_mild_demented = tf_utils.load_img(
    'dataset/AugmentedAlzheimerDataset/VeryMildDemented/000a074f-a3a5-4c70-8c94-d7ed7bbe7018.jpg', target_size = (64,64))

img_teste_very_mild_demented = tf_utils.img_to_array(img_teste_very_mild_demented)
img_teste_very_mild_demented /= 255
img_teste_very_mild_demented = np.expand_dims(img_teste_very_mild_demented, axis = 0)

# Teste Non Demented
img_teste_non_demented = tf_utils.load_img(
    'dataset/AugmentedAlzheimerDataset/NonDemented/0b02c1e3-4ac0-48d8-a28c-504618f3a7f6.jpg', target_size = (64,64))

img_teste_non_demented = tf_utils.img_to_array(img_teste_non_demented)
img_teste_non_demented /= 255
img_teste_non_demented = np.expand_dims(img_teste_non_demented, axis = 0)

# Teste Moderate Demented
img_teste_moderate_demented = tf_utils.load_img(
    'dataset/AugmentedAlzheimerDataset/ModerateDemented/0ef4d7f3-f245-4708-9f8d-26bddeec73e1.jpg', target_size = (64,64))

img_teste_moderate_demented = tf_utils.img_to_array(img_teste_moderate_demented)
img_teste_moderate_demented /= 255
img_teste_moderate_demented = np.expand_dims(img_teste_moderate_demented, axis = 0)

previsao = classificador.predict(img_teste_non_demented) 

