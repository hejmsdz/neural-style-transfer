import numpy as np
import matplotlib.pyplot as plt
from keras.applications.vgg19 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array

def imread(filename):
    img = load_img(filename, target_size=(224, 224))
    return img_to_array(img)

def imshow(image):
    plt.imshow(image / 255)
    plt.show()

def rgb2neural(image):
    return preprocess_input(image.copy())

def neural2rgb(image):
    out = image[..., ::-1].copy()
    mean = [123.68, 116.779, 103.939]
    out[..., 0] += mean[0]
    out[..., 1] += mean[1]
    out[..., 2] += mean[2]
    return out

def resize(image):
    target_size = 224
    dim = min(image.shape[:-1])
    image = image[0:dim, 0:dim]
    image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    return image
