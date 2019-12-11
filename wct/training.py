import glob
import itertools
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from .autoencoder import autoencoder
from .img import imread

def training_images(paths, batch_size, epochs=1):
    batches = [paths[pos : pos + batch_size] for pos in range(0, len(paths), batch_size)]
    for _ in range(epochs):
        for i, batch in enumerate(batches):
            images = np.array([imread(path) for path in batch])
            yield images, images # expected input and output for an auto-encoder are equal

def save_checkpoints():
    return ModelCheckpoint(
        'models/autoencoder.h5',
        monitor='loss',
        save_best_only=True,
        save_weights_only=True
    )

if __name__ == '__main__':
    autoencoder.compile(optimizer='adam', loss='mse')
    paths = glob.glob('images/*.png')
    epochs = 5
    batch_size = 3
    options = {
        'epochs': epochs,
        'steps_per_epoch': len(paths) / batch_size,
        'callbacks': [save_checkpoints()]
    }
    generator = training_images(paths, batch_size, epochs)
    autoencoder.fit_generator(generator, **options)
