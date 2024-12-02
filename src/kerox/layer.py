from collections.abc import Callable

import ndonnx
import spox
from numpy.typing import ArrayLike
from typing_extensions import Unpack

from kerox.core import EagerTensor, SupportsSpoxVar, SymbolicTensor
from kerox.namespace import register_object, to_unique


class Layer:
    def __init__(self, name: str = None):
        self._name = to_unique(name or self.__class__.__name__)
        register_object(self, self._name)
        self._source_layers = None
        self._eager_forward_function = None

    @property
    def built(self) -> bool:
        return True

    @property
    def eager_forward_function(
        self,
    ) -> Callable[[Unpack[tuple[SupportsSpoxVar, ...]]], ndonnx.Array]:
        if self._eager_forward_function is not None:
            return self._eager_forward_function

        from ndonnx._propagation import eager_propagate

        @eager_propagate
        def eager_forward_function(
            *inputs: Unpack[tuple[SupportsSpoxVar, ...]],
        ) -> ndonnx.Array:
            result = self.forward(*inputs)
            return ndonnx.from_spox_var(result)

        self._eager_forward_function = eager_forward_function
        return self._eager_forward_function

    def forward(self, *inputs: Unpack[tuple[SupportsSpoxVar, ...]]) -> spox.Var:
        raise NotImplementedError

    def __call__(
        self, inputs: SupportsSpoxVar | ArrayLike
    ) -> SymbolicTensor | EagerTensor:
        # Called with SymbolicTensor: set source layer and return SymbolicTensor
        if isinstance(inputs, SymbolicTensor):
            self._source_layers = [inputs.source_layer]
            result = self.forward(inputs)
            return SymbolicTensor(
                spox_var=result,
                shape=result.unwrap_tensor().shape,
                dtype=result.unwrap_tensor().dtype,
                source_layer=self._name,
            )

        # Called with EagerTensor or convetable to ndonnx.Array: return EagerTensor
        if isinstance(inputs, EagerTensor):
            inputs = inputs.value
        else:
            inputs = ndonnx.asarray(inputs)

        result = self.eager_forward_function(inputs)
        return EagerTensor(spox_var=result.spox_var())
