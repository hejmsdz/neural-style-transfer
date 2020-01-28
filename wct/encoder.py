from keras.layers import Input, Conv2D, MaxPooling2D
from keras.models import Model
from .unpooling import mask_make
from .utils import load_weights

def create_encoder(inputs, target_layer):
    """
    VGG19, up to the target layer (1 for relu1_1, 2 for relu2_1, etc.)
    """

    x = inputs

    masks = []
    print("BLOCK 1:")
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
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

    return x, masks

    # model = Model(inputs, x, name='vgg19', trainable=False)
    # load_weights(model)
    # return model, masks
