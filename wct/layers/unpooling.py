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

def mask_make(post_pooling, pre_pooling, name):
    upsampled = UpSampling2D()(post_pooling)
    _, a, b, c = pre_pooling.shape # TODO: make it work with fully convolutional architecture (unspecified input size)
    upsampled_flat = Reshape((1, a * b * c))(upsampled)
    pre_pooling_flat = Reshape((1, a * b * c))(pre_pooling)
    together = Concatenate(axis=-1)([pre_pooling_flat, upsampled_flat])
    tog_reshaped = Reshape((2, a, b, c))(together)

    bool_mask = Lambda(lambda t: K.greater_equal(t[:, 0], t[:, 1]), name=f"{name}_bool_mask")(tog_reshaped)
    mask = Lambda(lambda t: K.cast(t, dtype='float32'), name=f"{name}_float_mask")(bool_mask)
    return mask
