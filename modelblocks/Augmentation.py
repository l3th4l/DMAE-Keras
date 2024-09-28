import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
#from tensorflow.keras import ops

import numpy as np
import matplotlib.pyplot as plt
import random

#MAE training augmentation model 
def get_train_augmentation_model():
    model = keras.Sequential(
        [
            L.Rescaling(1 / 255.0),
            L.Resizing(INPUT_SHAPE[0] + 20, INPUT_SHAPE[0] + 20),
            L.RandomCrop(IMAGE_SIZE, IMAGE_SIZE),
            L.RandomFlip("horizontal"),
        ],
        name="train_data_augmentation",
    )
    return model


#MAE test augmentation model 
def get_test_augmentation_model():
    model = keras.Sequential(
        [L.Rescaling(1 / 255.0), L.Resizing(IMAGE_SIZE, IMAGE_SIZE),],
        name="test_data_augmentation",
    )
    return model

#Teacher Augmentation 
data_augmentation = keras.Sequential(
    [
        L.Normalization(),
        L.Resizing(IMAGE_SIZE, IMAGE_SIZE),
        L.RandomFlip("horizontal"),
        L.RandomRotation(factor=0.02),
        L.RandomZoom(height_factor=0.2, width_factor=0.2),
    ],
    name="data_augmentation",
)