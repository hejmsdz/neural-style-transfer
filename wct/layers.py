import h5py
import tensorflow as tf
from keras.layers import Lambda, Conv2D

from keras.layers import Input, Conv2D, UpSampling2D, Reshape, Concatenate
import keras
from keras.layers import Lambda
import keras.backend as K
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, Input
from keras.layers import Input, Conv2D, UpSampling2D
from keras.utils.data_utils import get_file
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.utils import plot_model

WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')


# def unpool(args):
#     mask, x = args
#     return keras.layers.multiply([mask, x])

def unpool(args):
    mask, x = args
    return keras.layers.multiply([mask, UpSampling2D()(x)])


def mask_make(x, orig):
    t = UpSampling2D()(x)
    _, a, b, c = orig.shape
    print("ORIGINAL:")
    print(a, b, c)
    xReshaped = Reshape((1, a * b * c))(t)
    origReshaped = Reshape((1, a * b * c))(orig)
    print("ENLARGED:")
    print(xReshaped.shape)
    print("ORIG ENLARGED:")
    print(origReshaped.shape)
    together = Concatenate(axis=-1)([origReshaped, xReshaped])
    togReshaped = Reshape((2, a, b, c))(together)
    print("TOG RESHAPED:")
    print(togReshaped.shape)

    bool_mask = Lambda(lambda t: K.greater_equal(t[:, 0], t[:, 1]))(togReshaped)

    mask = Lambda(lambda t: K.cast(t, dtype='float32'))(bool_mask)
    # mask = Reshape((a,b,c))(mask)
    print(mask.shape)
    return mask


def vgg_layers(inputs, target_layer):
    masks = []
    print("BLOCK 1:")
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    orig = x
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    masks.append(mask_make(x, orig))
    if target_layer == 1:
        return x, masks

    print("BLOCK 2:")
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    orig = x
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    masks.append(mask_make(x, orig))
    if target_layer == 2:
        return x, masks

    print("BLOCK 3:")
    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    orig = x
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    masks.append(mask_make(x, orig))
    if target_layer == 3:
        return x, masks

    print("BLOCK 4:")
    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    orig = x
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    masks.append(mask_make(x, orig))
    if target_layer == 4:
        return x, masks

    print("BLOCK 5:")
    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    orig = x
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    masks.append(mask_make(x, orig))

    return x, masks


def load_weights(model):
    weights_path = tf.keras.utils.get_file(
        'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
        WEIGHTS_PATH_NO_TOP,
        cache_subdir='models',
        file_hash='253f8cb515780f3b799900260a226db6')

    f = h5py.File(weights_path)

    layer_names = [name for name in f.attrs['layer_names']]

    for layer in model.layers:
        b_name = layer.name.encode()
        if b_name in layer_names:
            g = f[b_name]
            weights = [g[name] for name in g.attrs['weight_names']]
            layer.set_weights(weights)
            layer.trainable = False

    f.close()


def VGG19(input_tensor=None, input_shape=None, target_layer=1):
    """
    VGG19, up to the target layer (1 for relu1_1, 2 for relu2_1, etc.)
    """
    if input_tensor is None:
        inputs = Input(shape=input_shape)
    else:
        inputs = Input(tensor=input_tensor, shape=input_shape)

    layers, masks = vgg_layers(inputs, target_layer)
    model = Model(inputs, layers, name='vgg19')
    load_weights(model)
    return model, masks


def reflect(x, pad=1):
    return tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], 'REFLECT')


def ReflectingConv2D(*args, **kwargs):
    return Lambda(lambda x: Conv2D(*args, **kwargs)(reflect(x)), name=kwargs['name'])
