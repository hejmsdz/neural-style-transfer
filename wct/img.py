import cv2
import numpy as np

def imread(filename):
    return cv2.imread('images/valencia.png').astype(np.float32) / 255.0

def imshow(image, title='image'):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()