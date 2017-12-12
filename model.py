from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import UpSampling2D, GaussianNoise
from keras.layers import Activation, ELU, LeakyReLU
from keras.optimizers import Adam, RMSprop, SGD


def load_file_model(file):
    return load_model(file)


def mk_model():
    model = Sequential()

    model.add(GaussianNoise(0.05, input_shape=(256, 256, 3)))

    model.add(Conv2D(256, (3, 3), padding='same', name='conv-down-1'))
    model.add(BatchNormalization())
    model.add(ELU())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(GaussianNoise(0.01))

    model.add(Conv2D(256, (3, 3), padding='same', name='conv-down-2'))
    model.add(BatchNormalization())
    model.add(ELU())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same', name='conv-down-3'))
    model.add(BatchNormalization())
    model.add(ELU())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(GaussianNoise(0.05))

    model.add(Conv2D(256, (3, 3), padding='same', name='conv-middle'))
    model.add(BatchNormalization())
    model.add(ELU())

    model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same', name='conv-up-1'))
    model.add(BatchNormalization())
    model.add(ELU())

    model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same', name='conv-up-2'))
    model.add(BatchNormalization())
    model.add(ELU())

    model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same', name='conv-up-3'))
    model.add(BatchNormalization())
    model.add(ELU())
    model.add(Conv2D(3, (3, 3), padding='same', name='conv-dimreduc'))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))

    model.compile(loss='mse', optimizer=Adam(), metrics=['mae'])

    return model
