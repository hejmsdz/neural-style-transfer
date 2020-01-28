from keras.layers import MaxPooling2D, UpSampling2D, Reshape, Concatenate, Lambda, Multiply
import keras.backend as K

def MaskedMaxPooling2D(*args, **kwargs):
    def apply_layer(x):
        output = MaxPooling2D(*args, **kwargs)(x)
        mask = mask_make(output, x, kwargs['name'])
        return output, mask
    return apply_layer

def Unpooling2D(mask, *args, **kwargs):
    def apply_layer(x):
        x = UpSampling2D(*args, **kwargs)(x)
        x = Multiply(name=f"{kwargs['name']}_apply_mask")([mask, x])
        return x
    return apply_layer

def mask_make(x, orig, name):
    t = UpSampling2D()(x)
    _, a, b, c = orig.shape # TODO: make it work with fully convolutional architecture (unspecified input size)
    print("ORIGINAL:")
    print(a, b, c)
    xReshaped = Reshape((1, a * b * c))(t)
    origReshaped = Reshape((1, a * b * c))(orig)
    print("ENLARGED:")
    print(xReshaped.shape)
    print("ORIG ENLARGED:")
    print(origReshaped.shape)
    together = Concatenate(axis=-1, name=f"{name}_")([origReshaped, xReshaped])
    togReshaped = Reshape((2, a, b, c))(together)
    print("TOG RESHAPED:")
    print(togReshaped.shape)

    bool_mask = Lambda(lambda t: K.greater_equal(t[:, 0], t[:, 1]), name=f"{name}_bool_mask")(togReshaped)

    mask = Lambda(lambda t: K.cast(t, dtype='float32'), name=f"{name}_float_mask")(bool_mask)
    print(mask.shape)
    return mask

