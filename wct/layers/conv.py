import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Lambda

def reflect(x, pad=1):
    return tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], 'REFLECT')

def ReflectingConv2D(*args, **kwargs):
    def apply_layer(x):
        x = Lambda(reflect, name=f"{kwargs['name']}_reflect")(x)
        x = Conv2D(*args, **kwargs)(x)
        return x
    return apply_layer
