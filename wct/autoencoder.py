from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D
from tensorflow.keras.applications import VGG19 as encoder

# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, UpSampling2D, Lambda

e = encoder(include_top=False)
e.summary()

decoder = Sequential()


decoder.add(Input(shape=(None, None, 256)))

# hook into relu_3_1
# block 3
decoder.add(Conv2D(128, (3, 3), activation='relu'))
decoder.add(UpSampling2D(interpolation='nearest'))
decoder.add(Conv2D(128, (3, 3), activation='relu'))

# block 2
decoder.add(Conv2D(64, (3, 3), activation='relu'))
decoder.add(UpSampling2D(interpolation='nearest'))

# block 1
decoder.add(Conv2D(64, (3, 3), activation='relu'))

decoder.build()

decoder.summary()
