import h5py
import tensorflow as tf

WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

def load_vgg_weights(model):
    weights_path = tf.keras.utils.get_file(
        'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
        WEIGHTS_PATH_NO_TOP,
        cache_subdir='models',
        file_hash='253f8cb515780f3b799900260a226db6')

    with h5py.File(weights_path, 'r') as f:
        layer_names = [name for name in f.attrs['layer_names']]

        for layer in model.layers:
            b_name = layer.name.encode()
            if b_name in layer_names:
                g = f[b_name]
                weights = [g[name] for name in g.attrs['weight_names']]
                layer.set_weights(weights)
                layer.trainable = False
