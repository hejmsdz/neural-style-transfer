import cv2
import numpy as np
import matplotlib.pyplot as plt

def imread(filename):
    return cv2.imread(filename).astype(np.float32) / 255.0

def imshow(image, title='image'):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(rgb)
    plt.show()

def resize(image):
    target_size = 224
    dim = min(image.shape[:-1])
    image = image[0:dim, 0:dim]
    image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return image
