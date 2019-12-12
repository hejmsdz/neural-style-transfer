import glob
import sys
import cv2
from .img import imread, resize

paths = glob.glob(sys.argv[1])

for i, path in enumerate(paths):
    im = imread(path)
    im = resize(im)
    cv2.imwrite(f"images/train/{i}.jpg", (im * 255).astype(int))
