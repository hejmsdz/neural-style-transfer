import tensorflow as tf
from keras.layers import Conv2D, Lambda

def reflect(x, pad=1):
    return tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], 'REFLECT')

def ReflectingConv2D(*args, **kwargs):
    def reflect_convolve(x):
        x = reflect(x)
        x = Conv2D(*args, **kwargs)(x)
        return x
        
    return Lambda(reflect_convolve, name=kwargs['name'])
