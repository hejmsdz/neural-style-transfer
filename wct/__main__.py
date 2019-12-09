from PIL import Image
import numpy as np
from tensorflow.keras.applications import VGG19

im = Image.open('images/valencia.png')
pixels = np.array(im) / 255

encoder = VGG19()

features = encoder.predict(np.array([pixels]))
print(features[0])
