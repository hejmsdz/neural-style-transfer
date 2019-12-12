import cv2
import numpy as np

def imread(filename):
    return cv2.imread(filename).astype(np.float32) / 255.0

def imshow(image, title='image'):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize(image):
    target_size = 224
    dim = min(image.shape[:-1])
    image = image[0:dim, 0:dim]
    image = cv2.resize(image, (target_size, target_size))
    return image
