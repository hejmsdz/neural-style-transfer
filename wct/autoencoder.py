import keras
import numpy as np
from keras.layers import Lambda, Multiply, concatenate
from keras import Model, Sequential
from keras.layers import Input, UpSampling2D
from .layers import ReflectingConv2D, unpool, VGG19


def create_encoder(block):
    # output_layer = f"block{block}_conv1"
    # output = vgg.get_layer(output_layer).output
    # model = Model(vgg.input, output)
    # model.trainable = False
    return VGG19(input_shape=(224, 224, 3), target_layer=block)


def create_decoder(inputs_from_encoder, block, mask):
    input_depth = [None, 64, 128, 256, 512, 512][block]
    x = inputs_from_encoder

    if block == 5:
        x = ReflectingConv2D(512, (3, 3), activation='relu', name='dec_block5_conv4')(x)
        x = ReflectingConv2D(512, (3, 3), activation='relu', name='dec_block5_conv3')(x)
        x = ReflectingConv2D(512, (3, 3), activation='relu', name='dec_block5_conv2')(x)
        x = ReflectingConv2D(512, (3, 3), activation='relu', name='dec_block5_conv1')(x)
        # x = UpSampling2D((2, 2))(x)
        x = Lambda(unpool)([mask[4], x])

    if block >= 4:
        x = ReflectingConv2D(512, (3, 3), activation='relu', name='decoder_block4_conv4')(x)
        x = ReflectingConv2D(512, (3, 3), activation='relu', name='decoder_block4_conv3')(x)
        x = ReflectingConv2D(512, (3, 3), activation='relu', name='decoder_block4_conv2')(x)
        x = ReflectingConv2D(512, (3, 3), activation='relu', name='decoder_block4_conv1')(x)
        # x = UpSampling2D((2, 2))(x)
        x = Lambda(unpool)([mask[3], x])

    if block >= 3:
        x = ReflectingConv2D(256, (3, 3), activation='relu', name='decoder_block3_conv4')(x)
        x = ReflectingConv2D(256, (3, 3), activation='relu', name='decoder_block3_conv3')(x)
        x = ReflectingConv2D(256, (3, 3), activation='relu', name='decoder_block3_conv2')(x)
        x = ReflectingConv2D(256, (3, 3), activation='relu', name='decoder_block3_conv1')(x)
        # x = UpSampling2D((2, 2))(x)
        x = Lambda(unpool)([mask[2], x])

    if block >= 2:
        x = ReflectingConv2D(128, (3, 3), activation='relu', name='decoder_block2_conv2')(x)
        x = ReflectingConv2D(128, (3, 3), activation='relu', name='decoder_block2_conv1')(x)
        # x = UpSampling2D((2, 2))(x)
        x = Lambda(unpool)([mask[1], x])

    if block >= 1:
        x = ReflectingConv2D(64, (3, 3), activation='relu', name='decoder_block1_conv1')(x)
        x = ReflectingConv2D(64, (3, 3), activation='relu', name='decoder_block1_conv1')(x)
        # x = UpSampling2D((2, 2))(x)
        x = Lambda(unpool)([mask[0], x])

    # model = Sequential([
    #     Input(shape=(None, None, input_depth))
    # ])
    # for i in range(block, 0, -1):
    #     for layer in blocks[i]:
    #         unpooled_layer = keras.layers.multiply([mask[i - 1], layer])
    #         model.add(unpooled_layer)
    #

    model = Model(inputs=inputs_from_encoder, outputs=x)
    return model


def create_autoencoder(block):
    encoder, masks = create_encoder(block)
    decoder = create_decoder(encoder.output, block, masks)
    return encoder, decoder


# def chain_models(models):
#     first_input = models[0].input
#     last_output = models[0].output
#     for model in models[1:]:
#         last_output = model(last_output)
#     return Model(first_input, last_output)

def chain_models(models):
    return Sequential(models)


def transform(model, array):
    return model.predict(np.array([array]))[0]
