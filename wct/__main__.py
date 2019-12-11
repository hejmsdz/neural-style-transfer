import numpy as np
from .img import imread, imshow
from .autoencoder import autoencoder

im = imread('images/valencia.png')

out = autoencoder.predict(np.array([im]))[0]
print(out.shape)

imshow(out, 'autoencoder output')
