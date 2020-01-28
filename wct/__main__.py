import numpy as np
import glob
import cv2
import numpy as np
from tensorflow.keras.applications.vgg19 import preprocess_input

from .autoencoder import create_autoencoder, chain_models, transform
from .img import imread, imshow
from tensorflow.keras.applications.vgg19 import preprocess_input

from wct.wct import WCT
from .autoencoder import create_autoencoder, chain_models, transform
from .img import imread, imshow

wct = WCT()

rows = []
for style_path in glob.glob('styles/*.jpg'):
    style = imread(style_path)
    content = imread('lena.png')
    result = wct.stylize(style, content, block=2)
    rows.append(np.column_stack([style, result]))
results = np.row_stack(rows)
cv2.imwrite('results.png', results * 255)
imshow(results)

# encoder, decoder = create_autoencoder(3)
# autoencoder = chain_models([encoder, decoder])
# autoencoder.load_weights('models/decoder3.h5')
#
# img = imread('images/pasta.png')
# out = transform(autoencoder, img)
# print(out)
# imshow(out)

# fmap = transform(encoder, img)
# print(fmap)

# decoder.load_weights('models/decoder0.h5')
# rebuilt = transform(decoder, fmap)
# 
# print(rebuilt)
