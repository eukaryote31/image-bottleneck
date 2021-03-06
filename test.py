import model
from keras.preprocessing.image import array_to_img, img_to_array
from PIL import Image
import numpy as np
import sys
from load_images import get_data_noaug

job = sys.argv[1]

if job == 'meanimg':
    print "Computing mean image"

    data = get_data_noaug('crypto-icons')
    print "data shape", data.shape
    print "data type", data.dtype

    mpx = np.mean(data, axis=0)

    mpx *= 255
    mpx = mpx.astype(dtype='uint8')
    print "mean shape", mpx.shape
    print "mean type", mpx.dtype

    meanimg = Image.fromarray(mpx)
    meanimg.save('mean.png')

elif job == 'validate':
    model = model.load_file_model("model-anime.h5")
    i = img_to_array(Image.open(sys.argv[2]))
    i = i[:,:,:3]
    i = i.reshape((1,) + i.shape)
    i /= 256

    y = model.predict(i) * 255
    y = y.astype(dtype='uint8')

    predimg = Image.fromarray(y[0])
    predimg.save('res.png')
