from keras.layers import Input, Conv2D, UpSampling2D, Lambda
from keras.models import Model
from .unpooling import unpool
from .layers import ReflectingConv2D

def create_decoder(inputs, mask, block):
    # inputs = Input(shape=(14, 14, 512))

    x = inputs

    if block >= 4:
        x = ReflectingConv2D(512, (3, 3), activation='relu', name='decoder_block4_conv4')(x)
        x = ReflectingConv2D(512, (3, 3), activation='relu', name='decoder_block4_conv3')(x)
        x = ReflectingConv2D(512, (3, 3), activation='relu', name='decoder_block4_conv2')(x)
        x = ReflectingConv2D(512, (3, 3), activation='relu', name='decoder_block4_conv1')(x)
        x = UpSampling2D((2, 2))(x)
        x = Lambda(unpool)([x, mask[3]])

    if block >= 3:
        x = ReflectingConv2D(256, (3, 3), activation='relu', name='decoder_block3_conv4')(x)
        x = ReflectingConv2D(256, (3, 3), activation='relu', name='decoder_block3_conv3')(x)
        x = ReflectingConv2D(256, (3, 3), activation='relu', name='decoder_block3_conv2')(x)
        x = ReflectingConv2D(256, (3, 3), activation='relu', name='decoder_block3_conv1')(x)
        x = UpSampling2D((2, 2))(x)
        x = Lambda(unpool)([x, mask[2]])

    if block >= 2:
        x = ReflectingConv2D(128, (3, 3), activation='relu', name='decoder_block2_conv2')(x)
        x = ReflectingConv2D(128, (3, 3), activation='relu', name='decoder_block2_conv1')(x)
        x = UpSampling2D((2, 2))(x)
        x = Lambda(unpool)([x, mask[1]])

    if block >= 1:
        x = ReflectingConv2D(64, (3, 3), activation='relu', name='decoder_block1_conv2')(x)
        x = ReflectingConv2D(64, (3, 3), activation='relu', name='decoder_block1_conv1')(x)
        x = UpSampling2D((2, 2))(x)
        x = Lambda(unpool)([x, mask[0]])
    
    return x

    # model = Sequential([
    #     Input(shape=(None, None, input_depth))
    # ])
    # for i in range(block, 0, -1):
    #     for layer in blocks[i]:
    #         unpooled_layer = keras.layers.multiply([mask[i - 1], layer])
    #         model.add(unpooled_layer)
    #

    # return Model(inputs, x, name='decoder')
