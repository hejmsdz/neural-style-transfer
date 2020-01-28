from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from .encoder import encoder_layers
from .decoder import decoder_layers
from .utils import load_vgg_weights

def create_autoencoder(block):
    image = Input(shape=(224, 224, 3))
    encoded, masks = encoder_layers(block)(image)
    decoded = decoder_layers(block, masks)(encoded)
    autoencoder = Model(image, decoded)
    load_vgg_weights(autoencoder)
    return autoencoder

def transform(model, array):
    return model.predict(np.array([array]))[0]
