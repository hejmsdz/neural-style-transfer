from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D
from tensorflow.keras.applications import VGG19

from .layers import ReflectingConv2D

encoder = VGG19(include_top=False, input_shape=(224, 224, 3))
encoder.trainable = False
# encoder.summary()

layer_output = encoder.get_layer('block3_conv1').output

def make_decoder(input_layer):
    # block 3
    x = ReflectingConv2D(128, (3, 3), activation='relu', name='dec_block3_conv1')(input_layer)

    x = UpSampling2D(interpolation='nearest', name='dec_block3_upsample')(x)
    x = ReflectingConv2D(128, (3, 3), activation='relu', name='dec_block3_conv2')(x)

    # block 2
    x = ReflectingConv2D(64, (3, 3), activation='relu', name='dec_block2_conv1')(x)
    x = UpSampling2D(interpolation='nearest', name='dec_block2_upsample')(x)

    # block 1
    x = ReflectingConv2D(64, (3, 3), activation='relu', name='dec_block1_conv1')(x)

    # output
    x = ReflectingConv2D(3, 3, activation=None, name='dec_output')(x)
    return x

autoencoder = Model(encoder.input, outputs=[make_decoder(layer_output)])
autoencoder.build((224, 224))
autoencoder.summary()
