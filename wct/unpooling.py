from keras.layers import MaxPooling2D, UpSampling2D, Reshape, Concatenate, Lambda, Multiply
import keras.backend as K

def MaskedMaxPooling2D(*args, **kwargs):
    def apply_layer(x):
        output = MaxPooling2D(*args, **kwargs)(x)
        mask = mask_make(output, x)
        return output, mask
    return apply_layer

def mask_make(x, orig):
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
    together = Concatenate(axis=-1)([origReshaped, xReshaped])
    togReshaped = Reshape((2, a, b, c))(together)
    print("TOG RESHAPED:")
    print(togReshaped.shape)

    bool_mask = Lambda(lambda t: K.greater_equal(t[:, 0], t[:, 1]))(togReshaped)

    mask = Lambda(lambda t: K.cast(t, dtype='float32'))(bool_mask)
    # mask = Reshape((a,b,c))(mask)
    print(mask.shape)
    return mask

def unpool(args):
    x, input_mask = args
    # return keras.layers.multiply([input_mask, UpSampling2D()(x)])
    return Multiply()([input_mask, x])
