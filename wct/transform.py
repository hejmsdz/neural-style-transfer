import numpy as np
import os.path
import glob
import cv2
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from .img import imread, imshow
from .autoencoder import autoencoder

weights_file = 'models/autoencoder.h5'
if os.path.exists(weights_file):
    autoencoder.load_weights(weights_file)

content = imread('images/train/429.jpg')
style = imread('images/corridor.png')

encoder_output = autoencoder.get_layer('block3_conv1').output
encoder = Model(autoencoder.input, outputs=[encoder_output])
encoder.summary()

decoder_input = Input(encoder_output.shape)
autoencoder.get_layer('dec_block3_conv1').input = decoder_input
decoder = Model(decoder_input, outputs=[autoencoder.output])

decoder.summary()
