from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D
from tensorflow.keras.applications import VGG19

encoder = VGG19(include_top=False)
encoder.trainable = False
# encoder.summary()

layer_output = encoder.get_layer('block3_conv1').output

# block 3
x = Conv2D(128, (3, 3), activation='relu', name='dec_block3_conv1')(layer_output)
x = UpSampling2D(interpolation='nearest', name='dec_block3_upsample')(x)
x = Conv2D(128, (3, 3), activation='relu', name='dec_block3_conv2')(x)

# block 2
x = Conv2D(64, (3, 3), activation='relu', name='dec_block2_conv1')(x)
x = UpSampling2D(interpolation='nearest', name='dec_block2_upsample')(x)

# block 1
x = Conv2D(64, (3, 3), activation='relu', name='dec_block1_conv1')(x)
x = UpSampling2D(interpolation='nearest', name='dec_block1_upsample')(x)

autoencoder = Model(encoder.input, outputs=[layer_output, x])
autoencoder.build((224, 224))
autoencoder.summary()
