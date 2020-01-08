import numpy as np
from tensorflow.keras.applications import VGG19
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, UpSampling2D
from .layers import ReflectingConv2D

def create_encoder(block):
    vgg = VGG19(include_top=False)
    output_layer = f"block{block}_conv1"
    output = vgg.get_layer(output_layer).output
    model = Model(vgg.input, output)
    model.trainable = False
    return model

def create_decoder(block):
    input_depth = [None, 64, 128, 256, 512, 512][block]
    blocks = {
        5: [
            ReflectingConv2D(512, (3, 3), activation='relu', name='dec_block5_conv4'),
            UpSampling2D(interpolation='nearest', name='dec_block5_upsample'),
            ReflectingConv2D(512, (3, 3), activation='relu', name='dec_block5_conv3'),
            ReflectingConv2D(512, (3, 3), activation='relu', name='dec_block5_conv2'),
            ReflectingConv2D(512, (3, 3), activation='relu', name='dec_block5_conv1'),
        ],
        4: [
            ReflectingConv2D(256, (3, 3), activation='relu', name='dec_block4_conv4'),
            UpSampling2D(interpolation='nearest', name='dec_block4_upsample'),
            ReflectingConv2D(256, (3, 3), activation='relu', name='dec_block4_conv3'),
            ReflectingConv2D(256, (3, 3), activation='relu', name='dec_block4_conv2'),
            ReflectingConv2D(256, (3, 3), activation='relu', name='dec_block4_conv1'),
        ],
        3: [
            ReflectingConv2D(128, (3, 3), activation='relu', name='dec_block3_conv1'),
            UpSampling2D(interpolation='nearest', name='dec_block3_upsample'),
        ],
        2: [
            ReflectingConv2D(128, (3, 3), activation='relu', name='dec_block3_conv2'),
            ReflectingConv2D(64, (3, 3), activation='relu', name='dec_block2_conv1'),
            UpSampling2D(interpolation='nearest', name='dec_block2_upsample'),
        ],
        1: [
            ReflectingConv2D(64, (3, 3), activation='relu', name='dec_block1_conv1'),
            ReflectingConv2D(3, 3, activation='sigmoid', name='dec_output')
        ]
    }

    model = Sequential([
        Input(shape=(None, None, input_depth))
    ])
    for i in range(block, 0, -1):
        for layer in blocks[i]:
            model.add(layer)
    
    return model

def create_autoencoder(block):
    encoder = create_encoder(block)
    decoder = create_decoder(block)
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
