import numpy as np
from .autoencoder import create_autoencoder

from keras.utils import plot_model

for block in [2]:
    ae = create_autoencoder(block)
    ae.summary()

    plot_model(ae, to_file=f"model{block}.png")
    print(ae.predict(np.zeros((1, 224, 224, 3))))
