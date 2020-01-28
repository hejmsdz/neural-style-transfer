from keras.models import Model
from keras.layers import Input 
from .encoder import create_encoder
from .decoder import create_decoder

image = Input(shape=(224, 224, 3))
for block in [1,2,3,4]:
    encoded, masks = create_encoder(image, block)
    decoded = create_decoder(encoded, masks, block)
    ae = Model(image, decoded)
    ae.summary()
