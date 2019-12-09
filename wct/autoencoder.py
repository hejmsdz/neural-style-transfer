from tensorflow.keras.applications import VGG19 as encoder

# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, UpSampling2D, Lambda

e = encoder(include_top=False)
e.summary()
