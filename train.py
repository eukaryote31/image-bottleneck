import load_images
import model

from keras.models import save_model

TRAIN_EXISTING = False

if TRAIN_EXISTING:
    model = model.load_file_model('model-anime.h5')
else:
    model = model.mk_model()

x = load_images.get_data_noaug()
model.fit(x, x, epochs=1, batch_size=32, validation_split=0.1)
save_model(model, 'model-anime.h5')
