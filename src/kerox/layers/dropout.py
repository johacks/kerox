from keras import layers as klayers

from kerox.layers import layer
from kerox.ops.random import dropout


class Dropout(klayers.Dropout, layer.Layer):
    def __init__(self, rate, seed=None, **kwargs):
        super().__init__(rate, noise_shape=None, seed=seed, **kwargs)

    def call(self, inputs, training=False):
        if training and self.rate > 0:
            return dropout(inputs, self.rate, seed=self.seed_generator)
        return inputs
