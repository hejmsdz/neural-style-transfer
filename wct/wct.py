import glob
import cv2
import numpy as np

from wct.decoder import create_decoder
from wct.encoder import create_encoder
from tensorflow.keras import Model
from .autoencoder import create_autoencoder, transform
from .img import imread, imshow, rgb2neural, neural2rgb


class WCT:
    def __init__(self):
        self.blocks = [1, 2, 3, 4]
        self.autoencoders = []
        self.load_autoencoders()

    def stylize(self, style, content, block=1):
        style_features, _style_dims = WCT.flatten(self.encode(block, style))
        content_features, content_dims = WCT.flatten(self.encode(block, content))

        whitened_content_features = WCT.whitening_transform(content_features)
        stylized_features = WCT.coloring_transform(style_features, whitened_content_features)
        blended_stylized_features = WCT.blend(content_features, stylized_features)

        return self.decode(block, WCT.unflatten(blended_stylized_features, content_dims))

    def encode(self, block, image):
        array = rgb2neural(image)
        encoder = self.autoencoders[block - 1][0]
        return encoder(array)

    def decode(self, block, encoded, masks):
        decoder = self.autoencoders[block - 1][1]
        return decoder([encoded, masks])

    def load_autoencoder(self, block, weights_path):
        encoder = create_encoder(block)
        decoder = create_decoder(block)

        image = encoder.input
        encoded, *masks = encoder(image)
        decoded = decoder([encoded, *masks])
        outputs = [decoded]

        autoencoder = Model(image, outputs=outputs)
        autoencoder.load_weights(weights_path)

        return encoder, decoder

    def load_autoencoders(self):
        for block in self.blocks:
            encoder, decoder = self.load_autoencoder(block=block, weights_path=f"models/28-01-2020/decoder{block}.h5")
            self.autoencoders.append((encoder, decoder))

    def test_autoencoders(self):
        img = imread('lena.png')
        for block in self.blocks:
            feature_map, masks = self.encode(block, img)
            reconstruction = self.decode(block, feature_map, masks)
            imshow(neural2rgb(reconstruction))

    @staticmethod
    def whitening_transform(features):
        features -= features.mean(axis=1, keepdims=True)  # center (mean wrt. spatial dimension)
        product = np.dot(features, features.T) / (features.shape[1] - 1)  # why divide?
        Ec, wc, _ = np.linalg.svd(product)
        Dc = np.diag(wc)
        # assert((product - Ec.dot(Dc).dot(Ec.T) < 1e-4).all()) # Ec * Dc * Ec^T == features * features^T
        Dc_minus_half = np.linalg.inv(Dc) ** 0.5
        return Ec.dot(Dc_minus_half).dot(Ec.T).dot(features)

    @staticmethod
    def coloring_transform(style_features, whitened_content_features):
        mean = style_features.mean(axis=1, keepdims=True)  # center (mean wrt. spatial dimension)
        style_features -= mean

        product = np.dot(style_features, style_features.T) / (style_features.shape[1] - 1)
        Es, ws, _ = np.linalg.svd(product)
        Ds = np.diag(ws)
        return Es.dot(Ds).dot(Es.T).dot(whitened_content_features) + mean

    @staticmethod
    def blend(dry, wet, alpha=0.9):
        return alpha * wet + (1 - alpha) * dry

    @staticmethod
    def flatten(array):
        height, width, _depth = array.shape
        return array.transpose((2, 0, 1)).reshape(-1, width * height), (height, width)

    @staticmethod
    def unflatten(array, dims):
        return array.transpose().reshape((*dims, -1))


if __name__ == '__main__':
    wct = WCT()
    wct.test_autoencoders()
    #
    # rows = []
    # for style_path in glob.glob('styles/*.jpg'):
    #     style = imread(style_path)
    #     content = imread('lena.png')
    #     result = wct.stylize(style, content, block=2)
    #     rows.append(np.column_stack([style, result]))
    # results = np.row_stack(rows)
    # cv2.imwrite('results.png', results * 255)
    # imshow(results)
