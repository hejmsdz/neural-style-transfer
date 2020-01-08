import numpy as np
from tensorflow.keras.applications.vgg19 import preprocess_input
from .autoencoder import create_autoencoder, chain_models, transform
from .img import imread, imshow

encoder, decoder = create_autoencoder(3)
autoencoder = chain_models([encoder, decoder])
autoencoder.load_weights('models/decoder3.h5')

img = imread('images/pasta.png')
out = transform(autoencoder, img)
print(out)
imshow(out)

# fmap = transform(encoder, img)
# print(fmap)

# decoder.load_weights('models/decoder0.h5')
# rebuilt = transform(decoder, fmap)
# 
# print(rebuilt)
