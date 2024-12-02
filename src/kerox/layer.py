from collections.abc import Callable
from functools import wraps
from typing import Optional

import ndonnx
import spox
from numpy.typing import ArrayLike
from optree import PyTree
from typing_extensions import Unpack

from kerox.core import EagerTensor, SymbolicTensor
from kerox.namespace import register_object, to_unique
from kerox.typing import SupportsSpoxVar


class Layer:
    def __init__(self, name: str = None):
        self._name = to_unique(name or self.__class__.__name__)
        register_object(self, self._name)
        self._source_layers = None
        self._eager_forward_function = None
        self._last_inputs = self._last_outputs = None

    @property
    def built(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_layers(self) -> Optional[list[str]]:
        return self._source_layers

    @property
    def eager_forward_function(
        self,
    ) -> Callable[[PyTree[SupportsSpoxVar]], PyTree[ndonnx.Array]]:
        if self._eager_forward_function is not None:
            return self._eager_forward_function

        from ndonnx._propagation import eager_propagate

        @eager_propagate
        @wraps(self.forward)
        def eager_forward_function(*args, **kwargs):
            result = self.forward(*args, **kwargs)
            if isinstance(result, dict):
                return [
                    ndonnx.from_spox_var(value) for _, value in sorted(result.items())
                ]
            if isinstance(result, (list, tuple)):
                return [ndonnx.from_spox_var(value) for value in result]
            if isinstance(result, spox.Var):
                return ndonnx.from_spox_var(result)
            raise ValueError(
                f"Unsupported return type from forward method: {type(result)}"
            )

        self._eager_forward_function = eager_forward_function
        return self._eager_forward_function

    def forward(self, *inputs: Unpack[tuple[SupportsSpoxVar, ...]]) -> spox.Var:
        raise NotImplementedError

    def __call__(
        self, *inputs: Unpack[tuple[SupportsSpoxVar | ArrayLike, ...]]
    ) -> SymbolicTensor | EagerTensor:
        if len(inputs) == 0:
            raise ValueError("At least one input is required.")
        all_symbolic = all(type(input_) is SymbolicTensor for input_ in inputs)
        any_symbolic = any(type(input_) is SymbolicTensor for input_ in inputs)
        if not all_symbolic and any_symbolic:
            raise ValueError(
                "All inputs must be SymbolicTensor or ArrayLike, not a mix of both."
            )
        self._last_inputs = inputs

        # Called with SymbolicTensor: set source layer and return SymbolicTensor
        if all_symbolic:
            self._source_layers = [
                input_.source_layer
                for input_ in inputs
                if input_.source_layer is not None
            ]
            result = self.forward(*inputs)
            self._last_outputs = SymbolicTensor(
                spox_var=result, source_layer=self._name
            )
            return self._last_outputs

        # Called with EagerTensor or convertable to ndonnx.Array: return EagerTensor
        inputs = [
            input_.value if isinstance(input_, EagerTensor) else ndonnx.asarray(input_)
            for input_ in inputs
        ]

        result = self.eager_forward_function(*inputs)
        self._last_outputs = EagerTensor(result, eager_source=self._last_inputs)
        return self._last_outputs
