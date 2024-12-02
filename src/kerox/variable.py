from typing import Callable, Optional, Sequence, SupportsIndex, TypeAlias

import ndonnx
import numpy as np
import spox
import spox._future
from numpy.typing import ArrayLike, DTypeLike

ShapeLike: TypeAlias = SupportsIndex | Sequence[SupportsIndex]
Initializer: TypeAlias = Callable[[ShapeLike, DTypeLike], ArrayLike] | ArrayLike


class Variable:
    def __init__(
        self,
        name: str,
        shape: Optional[ShapeLike] = None,
        dtype: Optional[DTypeLike] = None,
        initializer: Optional[Initializer] = None,
        trainable: bool = True,
    ) -> None:
        self._name = name
        self._shape = shape
        self._dtype = dtype
        self._initializer = initializer
        self._value = self._get_initial_value()
        self._spox_value = spox.argument(spox._future.initializer(self._value))
        self._trainable = trainable

    @property
    def name(self) -> str:
        return self._name

    @property
    def shape(self) -> ShapeLike:
        return self._shape

    @property
    def dtype(self) -> DTypeLike:
        return self._dtype

    @property
    def value(self) -> ndonnx.Array:
        return self._value

    @property
    def trainable(self) -> bool:
        return self._trainable

    def _get_initial_value(self) -> ndonnx.Array:
        def initialize(shape: ShapeLike, dtype: DTypeLike) -> ArrayLike:
            if not callable(self._initializer):
                return self._initializer
            if shape is None or dtype is None:
                raise ValueError(
                    "Shape and dtype must be set if initializer is callable."
                )
            if self._initializer is None:
                return np.random.normal(size=shape).astype(dtype)
            return self._initializer(shape, dtype)

        value = initialize(self._shape, self._dtype)
        return ndonnx.asarray(value)
