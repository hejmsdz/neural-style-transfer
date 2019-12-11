from PIL import Image
import numpy as np
from .autoencoder import encoder

im = Image.open('images/valencia.png')
pixels = np.array(im) / 255

features = encoder.predict(np.array([pixels]))[0]
print(features)
print(features.shape)
