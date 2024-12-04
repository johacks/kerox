import typing

from keras import saving
from keras.src.models import Functional as KerasFunctional
from keras.src.models import Sequential as KerasSequential
from keras.src.models.model import Model as KerasModel
from keras.src.models.model import (
    Trainer,
    functional_init_arguments,
)

from kerox import ops
from kerox.layers import layer


# layer.Layer inheritance applies kerox Layer __call__ method modifications
# Rest just replaces Functional with KeroxFunctional
class KeroxModel(KerasModel, layer.Layer):
    def __new__(cls, *args, **kwargs):
        # Signature detection for usage of `Model` as a `Functional`
        if functional_init_arguments(args, kwargs) and cls == KeroxModel:
            return KeroxFunctional.__new__(KeroxFunctional, *args, **kwargs)
        return typing.cast(cls, super().__new__(cls))

    def __init__(self, *args, **kwargs):
        Trainer.__init__(self)

        # Signature detection for usage of a `Model` subclass
        # as a `Functional` subclass
        if functional_init_arguments(args, kwargs):
            inject_functional_model_class(self.__class__)
            KeroxFunctional.__init__(self, *args, **kwargs)
        else:
            layer.Layer.__init__(self, *args, **kwargs)


@saving.register_keras_serializable(package="kerox")
class KeroxFunctional(KerasFunctional, KeroxModel):
    def _convert_inputs_to_tensors(self, flat_inputs):
        converted = []
        for x, input in zip(flat_inputs, self._inputs):
            if x is None:  # TODO: check if optional
                converted.append(x)
            else:
                converted.append(
                    ops.convert_to_tensor(x, dtype=input.dtype, sparse=input.sparse)
                )
        return converted


@saving.register_keras_serializable(package="kerox")
class KeroxSequential(KerasSequential, KeroxModel):
    def build(self, input_shape=None):
        super().build(input_shape)
        if self._functional:  # Cast to KeroxFunctional by replacing methods
            self._functional._convert_inputs_to_tensors = (
                KeroxFunctional._convert_inputs_to_tensors
            )
            self._functional = typing.cast(KeroxFunctional, self._functional)


def inject_functional_model_class(cls):
    """Inject `Functional` into the hierarchy of this class if needed."""

    if cls is KeroxModel:
        return KeroxFunctional
    # In case there is any multiple inheritance, we stop injecting the
    # class if keras model is not in its class hierarchy.
    if cls is object:
        return object

    cls.__bases__ = tuple(inject_functional_model_class(base) for base in cls.__bases__)
    # Trigger any `__new__` class swapping that needed to happen on `Functional`
    # but did not because functional was not in the class hierarchy.
    cls.__new__(cls)

    return cls


__all__ = ["KeroxModel", "KeroxSequential"]
