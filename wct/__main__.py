import numpy as np
import os.path
import glob
from .img import imread, imshow
from .autoencoder import autoencoder

weights_file = 'models/autoencoder.h5'
if os.path.exists(weights_file):
    autoencoder.load_weights(weights_file)

for path in glob.glob('images/*.png'):
    im = imread(path)
    out = autoencoder.predict(np.array([im]))[0]
    imshow(np.column_stack([im, out]), 'autoencoder output')
