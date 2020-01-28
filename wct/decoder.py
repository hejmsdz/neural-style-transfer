from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Lambda
from tensorflow.keras import Model
from .layers.unpooling import Unpooling2D
from .layers.conv import ReflectingConv2D

def create_decoder(block):
    input_shape = [None, (112, 112, 64), (56, 56, 128), (28, 28, 256), (14, 14, 512)][block]
    mask_shapes = [(224, 224, 64), (112, 112, 128), (56, 56, 256), (28, 28, 512)]
    features = Input(shape=input_shape)
    masks = [Input(shape=mask_shapes[i]) for i in range(block)]
    decoded = decoder_layers(block, masks)(features)
    return Model(inputs=[features, *masks], outputs=[decoded], name='decoder')

def decoder_layers(block, masks):
    def apply_layers(x):
        if block >= 4:
            x = ReflectingConv2D(512, (3, 3), activation='relu', name='decoder_block4_conv4')(x)
            x = ReflectingConv2D(512, (3, 3), activation='relu', name='decoder_block4_conv3')(x)
            x = ReflectingConv2D(512, (3, 3), activation='relu', name='decoder_block4_conv2')(x)
            x = ReflectingConv2D(512, (3, 3), activation='relu', name='decoder_block4_conv1')(x)
            x = Unpooling2D(masks[3], (2, 2), name='decoder_block4_unpool')(x)

        if block >= 3:
            x = ReflectingConv2D(256, (3, 3), activation='relu', name='decoder_block3_conv4')(x)
            x = ReflectingConv2D(256, (3, 3), activation='relu', name='decoder_block3_conv3')(x)
            x = ReflectingConv2D(256, (3, 3), activation='relu', name='decoder_block3_conv2')(x)
            x = ReflectingConv2D(256, (3, 3), activation='relu', name='decoder_block3_conv1')(x)
            x = Unpooling2D(masks[2], (2, 2), name='decoder_block3_unpool')(x)

        if block >= 2:
            x = ReflectingConv2D(128, (3, 3), activation='relu', name='decoder_block2_conv2')(x)
            x = ReflectingConv2D(128, (3, 3), activation='relu', name='decoder_block2_conv1')(x)
            x = Unpooling2D(masks[1], (2, 2), name='decoder_block2_unpool')(x)

        if block >= 1:
            x = ReflectingConv2D(64, (3, 3), activation='relu', name='decoder_block1_conv2')(x)
            x = ReflectingConv2D(64, (3, 3), activation='relu', name='decoder_block1_conv1')(x)
            x = Unpooling2D(masks[0], (2, 2), name='decoder_block1_unpool')(x)
        
        x = ReflectingConv2D(3, 3, activation=None, name='decoder_output')(x)
        return x
    
    return apply_layers
