import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from .unpooling import Unpooling2D, MaskedMaxPooling2D

def create_model():
    inputs = Input(shape=(4, 4, 1))
    pooled, mask = MaskedMaxPooling2D((2, 2), strides=(2, 2), name='pool')(inputs)
    unpooled = Unpooling2D(mask, (2, 2), name='unpool')(pooled)
    return Model(inputs, outputs=[pooled, mask, unpooled])

model = create_model()
data = np.array([
  [[3], [5], [1], [2]],
  [[4], [3], [0], [3]],
  [[0], [4], [3], [6]],
  [[7], [2], [2], [5]],
])
outputs = model.predict(np.array([data]))
pooled, mask, unpooled = [out[0] for out in outputs]

np.testing.assert_array_equal(pooled, np.array([
    [[5], [3]],
    [[7], [6]],
]))

np.testing.assert_array_equal(mask, np.array([
    [[0], [1], [0], [0]],
    [[0], [0], [0], [1]],
    [[0], [0], [0], [1]],
    [[1], [0], [0], [0]],
]))

np.testing.assert_array_equal(unpooled, np.array([
    [[0], [5], [0], [0]],
    [[0], [0], [0], [3]],
    [[0], [0], [0], [6]],
    [[7], [0], [0], [0]],
]))

print('Test passed')
