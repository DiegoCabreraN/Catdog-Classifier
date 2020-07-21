import os
import tensorflow as tf
import numpy as np

from PIL import Image
from tensorflow import keras as K

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def predict(image_path):
    model = K.models.Sequential([
        K.layers.Conv2D(32, 2, activation = 'relu',
                        kernel_regularizer = K.regularizers.l2(0.0001),
                        input_shape = (160, 160, 3)),
        K.layers.BatchNormalization(),
        K.layers.MaxPool2D(2,2),
        K.layers.Dropout(0.2),
        
        K.layers.Conv2D(64, 2, activation = 'relu', kernel_regularizer = K.regularizers.l2(0.0001)),
        K.layers.BatchNormalization(),
        K.layers.MaxPool2D(2,2),
        K.layers.Dropout(0.2),
        
        K.layers.Conv2D(128, 2, activation = 'relu',kernel_regularizer = K.regularizers.l2(0.0001)),
        K.layers.BatchNormalization(),
        K.layers.MaxPool2D(2,2),
        K.layers.Dropout(0.2),
        
        K.layers.Conv2D(256, 2, activation = 'relu', kernel_regularizer = K.regularizers.l2(0.0001)),
        K.layers.BatchNormalization(),
        K.layers.MaxPool2D(2,2),
        K.layers.Dropout(0.2),
        
        K.layers.Conv2D(512, 2, activation = 'relu', kernel_regularizer = K.regularizers.l2(0.0001)),
        K.layers.BatchNormalization(),
        K.layers.MaxPool2D(2,2),
        K.layers.Dropout(0.2),
        
        K.layers.Conv2D(1024, 2, activation = 'relu', kernel_regularizer = K.regularizers.l2(0.0001)),
        K.layers.BatchNormalization(),
        K.layers.MaxPool2D(2,2),
        K.layers.Dropout(0.2),
        
        K.layers.Flatten(),
        K.layers.Dense(512, activation = 'relu', kernel_regularizer = K.regularizers.l2(0.0001)),
        K.layers.BatchNormalization(),
        K.layers.Dropout(0.2),
        K.layers.Dense(1, activation='sigmoid'),
    ])
    model.trainable = False
    model.load_weights('model/model.h5')
    image = np.array(Image.open(image_path).resize((160,160), resample = 3))
    image = np.expand_dims(image, axis = 0)
    image = image / 255.0
    prediction = model.predict(image)[0][0]
    if prediction > 0.5:
        return 'Dog'
    return 'Cat'
