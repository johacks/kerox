from keras.src.models import Functional as KerasFunctionalModel

from kerox.core import KeroxTensor
from kerox.layers import Layer


class Functional(KerasFunctionalModel, Layer):
    def _standardize_inputs(self, inputs):
        if isinstance(inputs, KeroxTensor):
            return (inputs,)
        if all(isinstance(input_, KeroxTensor) for input_ in inputs):
            return (inputs,)
        return super()._standardize_inputs(inputs)
