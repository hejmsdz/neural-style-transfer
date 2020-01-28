import glob
import numpy as np
import matplotlib.pyplot as plt 
from keras.callbacks import ModelCheckpoint
from .autoencoder import create_autoencoder
from .img import imread, imshow, rgb2neural, neural2rgb

def training_images(paths, batch_size, epochs=1):
    batches = [paths[pos : pos + batch_size] for pos in range(0, len(paths), batch_size)]
    for _ in range(epochs):
        for batch in batches:
            images = np.array([imread(path) for path in batch])
            images = rgb2neural(images)
            yield images

def validation_images(paths):
    return list(training_images(paths, batch_size=len(paths), epochs=1))

def draw_learning_curve(history, block):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.grid()
    plt.title(f"Autoencoder loss at block {block}")
    plt.xlabel('epoch')
    plt.ylabel('content + feature mse')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.show()

def test_autoencoder(img_path):
    img = imread(img_path)
    imgs = np.expand_dims(img, 0)
    imgs = rgb2neural(imgs)
    rebuilt = ae1.predict(imgs)[0]
    rebuilt = neural2rgb(rebuilt)
    imshow(np.column_stack([img, rebuilt]))

def save_checkpoints(block, path):
    return ModelCheckpoint(
        f"{path}/decoder{block}.h5",
        monitor='loss',
        save_best_only=True,
        save_weights_only=True
    )

def train(block, train_path, valid_path=None, weights_path='models', epochs=10):
    print(f"Training decoder at block {block}")
    autoencoder = create_autoencoder(block, reencode=True)
    autoencoder.compile(optimizer='adam')
    train_files = glob.glob(train_path)
    valid_files = validation_images(glob.glob(valid_path)) if valid_path else None
    batch_size = 10
    options = {
        'epochs': epochs,
        'steps_per_epoch': len(train_files) / batch_size,
        'callbacks': [save_checkpoints(block, weights_path)],
        'validation_data': valid_files,
    }
    generator = training_images(train_files, batch_size, epochs)
    history = autoencoder.fit(generator, **options)
    return autoencoder, history

if __name__ == '__main__':
    train_path = 'images/train/*.jpg'
    valid_path = None
    weights_path = 'models/2020-01-28'
    for i in [1, 2]:
        train(i, train_path, valid_path, weights_path)
