from keras.layers import Input, Conv2D, MaxPooling2D
from keras.models import Model
from .layers.unpooling import MaskedMaxPooling2D

def encoder_layers(block):
    def apply_layers(x):
        masks = []

        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x, mask = MaskedMaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
        masks.append(mask)

        if block == 1:
            return x, masks

        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x, mask = MaskedMaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
        masks.append(mask)

        if block == 2:
            return x, masks

        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
        x, mask = MaskedMaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
        masks.append(mask)

        if block == 3:
            return x, masks

        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
        x, mask = MaskedMaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
        masks.append(mask)

        if block == 4:
            return x, masks

    return apply_layers
