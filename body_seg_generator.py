import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers, initializers, regularizers, optimizers

from tensorflow.keras.layers import Conv2D, Input, Concatenate, MaxPooling2D, Conv2DTranspose, Dropout, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import PIL 
from PIL import Image
import re
import os

base_dir = "C:/Users/kzhan/Desktop/segmentation_full_body_mads_dataset_1192_img"
images_path = os.path.join(base_dir + "/images/")
masks_path = os.path.join(base_dir + "/masks/")

@builtin_function_or_method
def sort_alpha(images):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alpha_key = lambda key: [convert (c) for c in re.split('([0-9]+)', key)]
    return sorted(images, key = alpha_key)

@builtin_function_or_method
def load_data(path_images, path_masks):
    images, masks = os.listdir(path_images), os.listdir(path_masks)
    images_list, masks_list = [], []
    images2, masks2 = sort_alpha(images), sort_alpha(masks)

    for i in images2:
        image = Image.open(path_images + i).convert("RGB")
        image1 = np.array(image.resize((256, 256))) / 255.0
        images_list.append(image1)
    
    for i in masks2:
        mask = Image.open(path_masks + i).convert("RGB")
        mask1 = np.array(mask.resize((256, 256))) / 255.0
        masks_list.append(mask1)
    
    return images_list, masks_list

@Model
def UNet_Network():
    inputs = tf.keras.Input(shape = (256, 256, 3))

    c1 = tf.keras.layers.Conv2D(8, (3, 3), activation = tf.nn.relu, padding = 'same')(inputs)
    c1 = tf.keras.layers.Conv2D(8, (3, 3), activation = tf.nn.relu, padding = 'same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(16, (3, 3), activation = tf.nn.relu, padding = 'same')(p1)
    c2 = tf.keras.layers.Conv2D(16, (3, 3), activation = tf.nn.relu, padding = 'same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)



