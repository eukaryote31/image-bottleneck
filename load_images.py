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

def get_data_noaug(directory='thumbnail/', keep_grayscale=False):
    cases = []
    for f in os.listdir(directory):
        im = Image.open(directory + "/" + f)
        pixels = np.array(im, dtype='float32')

        if len(pixels.shape) == 2:
            if not keep_grayscale:
                continue
            pixels = pixels.reshape(pixels.shape + (1,))

        try:
            # cut off Alpha channel
            if pixels.shape[-1] != 1:
                pixels = pixels[:,:,:-1]
        except:
            print f
        # drop grayscale images
        if pixels.shape[-1] == 1:
            if not keep_grayscale:
                continue
            pixels = np.reshape(np.stack((pixels,) * 3, axis=-1), pixels.shape[:-1] + (3,))


        # scale to [0..1]
        pixels /= 256

        cases.append(pixels)
        cases.append(np.fliplr(pixels))

    return np.array(cases)
