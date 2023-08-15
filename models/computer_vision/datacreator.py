import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from constants import *

class DataCreator:
    def train_generators(self, validation_split = 0.1, batch_size = 8):
        train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                           validation_split=validation_split,
                                           horizontal_flip=True,
                                           dtype='float16')
        
        # Constant seed is required to avoir mixing train and dev data 
        # while keeping the same model when excecuting the program multiple times
        
        train_generator = train_datagen.flow_from_directory(
            'data/train/',
            target_size=(IMG_SIZE,IMG_SIZE),
            color_mode='rgb',
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True,
            subset='training',
            seed = 1)
        
        val_generator = train_datagen.flow_from_directory(
            'data/train/',
            target_size=(IMG_SIZE,IMG_SIZE),
            color_mode='rgb',
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True,
            subset='validation',
            seed = 1)    
        
        return train_generator,val_generator 
    
    def predict_generator(self):
        pred_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, dtype='float16')    
        test_generator = pred_datagen.flow_from_directory(
            directory= 'data/test/',
            target_size=(IMG_SIZE, IMG_SIZE),
            color_mode="rgb",
            shuffle = False,
            class_mode='categorical',
            batch_size=1,
            seed = 1)
        return test_generator
    
