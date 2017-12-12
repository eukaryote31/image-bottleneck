from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os
from PIL import Image
import numpy as np


def get_data():
    train_datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.05,
            height_shift_range=0.05,
            rescale=1./255,
            shear_range=0.05,
            zoom_range=0.05,
            horizontal_flip=True,
            fill_mode='nearest')

    test_datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            rescale=1./255,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest')

    train = train_datagen.flow_from_directory('./images/train', target_size=(256, 256), batch_size=32, class_mode='input')

    test = test_datagen.flow_from_directory('./images/test', target_size=(256, 256), batch_size=32, class_mode='input')

    return train, test

def get_data_noaug():
    cases = []
    for f in os.listdir("thumbnail"):
        im = Image.open("thumbnail/" + f)
        pixels = np.array(im, dtype='float32')

        # cut off Alpha channel
        pixels = pixels[:,:,:-1]

        # scale to [0..1]
        pixels /= 256

        # drop grayscale images
        if pixels.shape == (256, 256, 1):
            pixels = np.stack((pixels,) * 3)
        cases.append(pixels)
#        cases.append(np.fliplr(pixels))

    return np.array(cases)
