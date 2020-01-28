from tensorflow.keras.layers import Input
from tensorflow.keras import Model
from tensorflow.keras.losses import mean_squared_error
import tensorflow.keras.backend as K
from .encoder import create_encoder
from .decoder import create_decoder
from .utils import load_vgg_weights

def create_autoencoder(block, reencode=False):
    encoder = create_encoder(block)
    decoder = create_decoder(block)

    image = encoder.input
    encoded, *masks = encoder(image)
    decoded = decoder([encoded, *masks])
    outputs = [encoded, decoded]

    if reencode:
        encoder2 = create_encoder(block)
        reencoded, *_masks = encoder(decoded) # take image, discard masks
        outputs.append(reencoded)

    autoencoder = Model(image, outputs=outputs)
    if reencode:
        autoencoder.add_loss(content_feature_loss(image, encoded, decoded, reencoded))
    return autoencoder

def content_feature_loss(content1, features1, content2, features2, feature_weight=1.0):
    content_mse = mean_squared_error(content1, content2)
    features_mse = mean_squared_error(features1, features2)
    return K.mean(content_mse) + feature_weight * K.mean(features_mse)

def transform(model, array):
    return model.predict(np.array([array]))[0]
