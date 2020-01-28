from keras.models import Model
from keras.layers import Input 
from .encoder import encoder_layers
from .decoder import decoder_layers

image = Input(shape=(224, 224, 3))
for block in [1,2,3,4]:
    encoded, masks = encoder_layers(image, block)
    decoded = decoder_layers(encoded, masks, block)
    ae = Model(image, decoded)
    ae.summary()
