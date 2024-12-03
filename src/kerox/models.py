from typing import Sequence

from keras import saving
from keras.src.models import Functional as KerasFunctionalModel

from kerox.core import KeroxTensor
from kerox.layers import Layer


@saving.register_keras_serializable(package="kerox")
class KeroxFunctional(KerasFunctionalModel, Layer):
    def _standardize_inputs(self, inputs):
        if isinstance(inputs, KeroxTensor):
            return (inputs,)
        if isinstance(inputs, Sequence):
            if all(isinstance(input_, KeroxTensor) for input_ in inputs):
                return inputs
        return super()._standardize_inputs(inputs)
