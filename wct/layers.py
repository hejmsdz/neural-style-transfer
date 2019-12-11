import tensorflow as tf
from tensorflow.keras.layers import Lambda, Conv2D

def reflect(x, pad=1):
    return tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], 'REFLECT')

def ReflectingConv2D(*args, **kwargs):
    return Lambda(lambda x: Conv2D(*args, **kwargs)(reflect(x)), name=kwargs['name'])
