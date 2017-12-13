from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import UpSampling2D, GaussianNoise, Input
from keras.layers import Activation, ELU, LeakyReLU
from keras.optimizers import Adam, RMSprop, SGD
from keras.models import Model


def load_file_model(file):
    return load_model(file)


def mk_model():

    inp = Input(shape=(256, 256, 3))
    l = GaussianNoise(0.05)(inp)

    l = Conv2D(256, (3, 3), padding='same', name='conv-down-1')(l)
    l = BatchNormalization()(l)
    l = ELU()(l)

    l = MaxPooling2D(pool_size=(2, 2))(l)

    l = GaussianNoise(0.01)(l)

    l = Conv2D(256, (3, 3), padding='same', name='conv-down-2')(l)
    l = BatchNormalization()(l)
    l = ELU()(l)

    l = MaxPooling2D(pool_size=(2, 2))(l)

    l = Conv2D(256, (3, 3), padding='same', name='conv-down-3')(l)
    l = BatchNormalization()(l)
    l = ELU()(l)

    l = MaxPooling2D(pool_size=(2, 2))(l)

    l = GaussianNoise(0.05)(l)

    l = Conv2D(256, (3, 3), padding='same', name='conv-middle')(l)
    l = BatchNormalization()(l)
    l = ELU()(l)

    l = UpSampling2D(size=(2, 2))(l)

    l = Conv2D(256, (3, 3), padding='same', name='conv-up-1')(l)
    l = BatchNormalization()(l)
    l = ELU()(l)

    l = UpSampling2D(size=(2, 2))(l)

    l = Conv2D(256, (3, 3), padding='same', name='conv-up-2')(l)
    l = BatchNormalization()(l)
    l = ELU()(l)

    l = UpSampling2D(size=(2, 2))(l)

    l = Conv2D(256, (3, 3), padding='same', name='conv-up-3')(l)
    l = BatchNormalization()(l)
    l = ELU()(l)
    l = Conv2D(3, (3, 3), padding='same', name='conv-dimreduc')(l)
    l = BatchNormalization()(l)
    l = Activation('sigmoid')(l)

    model = Model(inputs=[inp], outputs=[l])
    model.compile(loss='mse', optimizer=Adam(), metrics=['mae'])

    return model
