from typing import Optional

import spox
import spox.opset.ai.onnx.v21 as op

from kerox.core import Parameter
from kerox.layer import Layer
from kerox.typing import Activation, Initializer, SupportsSpoxVar


class Dense(Layer):
    def __init__(
        self,
        units: int,
        activation: Optional[Activation] = None,
        name: str = None,
        kernel_initializer: Optional[Initializer] = None,
        bias_initializer: Optional[Initializer] = None,
    ):
        super().__init__(name=name or "dense")
        self._units = units
        self._activation = activation
        self._kernel = None
        self._bias = None
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer

    def build(self, inputs: SupportsSpoxVar) -> None:
        if self.built:
            raise RuntimeError("Dense layer is built already.")
        if len(inputs.shape) != 2:
            raise ValueError("Dense inputs must have rank 2.")

        tensor = inputs.spox_var().unwrap_tensor()

        self._kernel = Parameter(
            "kernel",
            source_layer=self._name,
            shape=(tensor.shape[-1], self._units),
            dtype=tensor.dtype,
            initializer=self._kernel_initializer,
        )
        self._bias = Parameter(
            "bias",
            source_layer=self._name,
            shape=(self._units,),
            dtype=tensor.dtype,
            initializer=self._bias_initializer,
        )

    @property
    def built(self) -> bool:
        return self._kernel is not None and self._bias is not None

    def forward(self, inputs: SupportsSpoxVar) -> spox.Var:
        if not self.built:
            self.build(inputs)
        if len(inputs.shape) != 2:
            raise ValueError("Dense inputs must have rank 2.")

        # Matrix multiplication
        x = op.matmul(inputs.spox_var(), self._kernel.spox_var())
        x = op.add(x, self._bias.spox_var())
        if self._activation is not None:
            x = self._activation(x)
        return x

    def __repr__(self) -> str:
        return f"Dense(units={self._units}, activation={self._activation}, name={self._name})"
