import spox
from typing_extensions import Unpack

from kerox.layer import Layer
from kerox.typing import SupportsSpoxVar


class Sequential(Layer):
    def __init__(self, layers: list[Layer], name: str = None):
        super().__init__(name)
        if not layers:
            raise ValueError("At least one layer is required.")
        if not all(isinstance(layer, Layer) for layer in layers):
            raise ValueError("All layers must be instances of Layer.")
        self._layers = layers

    @property
    def layers(self) -> list[Layer]:
        return self._layers

    def forward(self, *inputs: Unpack[tuple[SupportsSpoxVar, ...]]) -> spox.Var:
        x = inputs
        for layer in self._layers:
            x = layer(x)
        return x
