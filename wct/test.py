import numpy as np
from keras.models import Model
from keras.layers import Input 
from .encoder import encoder_layers
from .decoder import decoder_layers

image = Input(shape=(224, 224, 3))
for block in [1,2,3,4]:
    encoded, masks = encoder_layers(block)(image)
    decoded = decoder_layers(block, masks)(encoded)
    ae = Model(image, decoded)
    ae.summary()
    print(ae.predict(np.zeros((1, 224, 224, 3))))
