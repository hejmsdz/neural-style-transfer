from keras.layers import Input, Conv2D, UpSampling2D, Lambda
from keras.models import Model
from .layers.unpooling import Unpooling2D
from .layers.conv import ReflectingConv2D

def decoder_layers(block, masks):
    def apply_layers(x):
        if block >= 4:
            x = ReflectingConv2D(512, (3, 3), activation='relu', name='decoder_block4_conv4')(x)
            x = ReflectingConv2D(512, (3, 3), activation='relu', name='decoder_block4_conv3')(x)
            x = ReflectingConv2D(512, (3, 3), activation='relu', name='decoder_block4_conv2')(x)
            x = ReflectingConv2D(512, (3, 3), activation='relu', name='decoder_block4_conv1')(x)
            x = Unpooling2D(masks[3], (2, 2))(x)

        if block >= 3:
            x = ReflectingConv2D(256, (3, 3), activation='relu', name='decoder_block3_conv4')(x)
            x = ReflectingConv2D(256, (3, 3), activation='relu', name='decoder_block3_conv3')(x)
            x = ReflectingConv2D(256, (3, 3), activation='relu', name='decoder_block3_conv2')(x)
            x = ReflectingConv2D(256, (3, 3), activation='relu', name='decoder_block3_conv1')(x)
            x = Unpooling2D(masks[2], (2, 2))(x)

        if block >= 2:
            x = ReflectingConv2D(128, (3, 3), activation='relu', name='decoder_block2_conv2')(x)
            x = ReflectingConv2D(128, (3, 3), activation='relu', name='decoder_block2_conv1')(x)
            x = Unpooling2D(masks[1], (2, 2))(x)

        if block >= 1:
            x = ReflectingConv2D(64, (3, 3), activation='relu', name='decoder_block1_conv2')(x)
            x = ReflectingConv2D(64, (3, 3), activation='relu', name='decoder_block1_conv1')(x)
            x = Unpooling2D(masks[0], (2, 2))(x)
        return x
    
    return apply_layers
