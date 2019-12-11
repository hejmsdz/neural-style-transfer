import numpy as np
import os.path
from .img import imread, imshow
from .autoencoder import autoencoder

im = imread('images/cookies.png')

weights_file = 'models/autoencoder.h5'
if os.path.exists(weights_file):
    autoencoder.load_weights(weights_file)

out = autoencoder.predict(np.array([im]))[0]
print(out.shape)

imshow(out, 'autoencoder output')
