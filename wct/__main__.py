import numpy as np
import cv2
from tensorflow.keras.applications import VGG19

img = cv2.imread('images/valencia.png')
encoder = VGG19()

features = encoder.predict(np.array([img]))
print(features[0])
