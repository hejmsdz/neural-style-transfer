import glob
import numpy as np
from keras.callbacks import ModelCheckpoint
from .autoencoder import create_autoencoder, chain_models
from .img import imread, resize

def training_images(paths, batch_size, epochs=1):
    batches = [paths[pos : pos + batch_size] for pos in range(0, len(paths), batch_size)]
    for _ in range(epochs):
        for batch in batches:
            images = np.array([imread(path) for path in batch])
            yield images, images

def save_checkpoints(block):
    return ModelCheckpoint(
        f"models/26-01-2020/decoder{block}.h5",
        monitor='loss',
        save_best_only=True,
        save_weights_only=True
    )

def train(block):
    print(f"Training decoder at block {block}")
    encoder, decoder = create_autoencoder(block)
    # encoder.build((224, 224, 3))
    # decoder.build((56, 56, 256))
    autoencoder = chain_models([encoder, decoder])

    autoencoder.compile(optimizer='adam', loss='mse')
    paths = glob.glob('images/train/*.jpg')
    epochs = 10
    batch_size = 10
    options = {
        'epochs': epochs,
        'steps_per_epoch': len(paths) / batch_size,
        'callbacks': [save_checkpoints(block)],
    }
    generator = training_images(paths, batch_size, epochs)
    autoencoder.fit_generator(generator, **options)

if __name__ == '__main__':
    for i in [5]:
        train(i)
    