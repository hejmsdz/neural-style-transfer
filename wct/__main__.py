from PIL import Image
import numpy as np
from .autoencoder import autoencoder

im = Image.open('images/valencia.png')
pixels = np.array(im) / 255

features = autoencoder.predict(np.array([pixels]))[0]
print(features)

im2 = Image.fromarray(np.uint8(features[0] * 255)[0])
im2.show()

print(features.shape)
