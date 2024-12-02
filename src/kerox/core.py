from typing import Optional

import ndonnx
import numpy as np
import spox
import spox._future
from numpy.typing import ArrayLike, DTypeLike, NDArray

from kerox.namespace import to_unique
from kerox.typing import Initializer, ShapeLike, SupportsSpoxVar


def default_initializer(shape: ShapeLike, dtype: DTypeLike) -> ArrayLike:
    return np.random.normal(size=shape).astype(dtype)


class Parameter(SupportsSpoxVar):
    def __init__(
        self,
        name: str,
        source_layer: Optional[str] = None,
        shape: Optional[ShapeLike] = None,
        dtype: Optional[DTypeLike] = None,
        initializer: Optional[Initializer] = None,
        trainable: bool = True,
    ) -> None:
        self._name = (
            f"{source_layer}.{name}" if source_layer is not None else to_unique(name)
        )
        self._shape = shape
        self._dtype = dtype
        self._initializer = initializer
        self._initial_value = self._get_initial_value()
        self._spox_var = spox._future.initializer(self._initial_value)
        self._spox_var._rename(self._name)
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
    def trainable(self) -> bool:
        return self._trainable

    @property
    def initial_value(self) -> NDArray:
        return self._initial_value

    def spox_var(self) -> spox.Var:
        return self._spox_var

    def _get_initial_value(self) -> NDArray:
        def initialize(shape: ShapeLike, dtype: DTypeLike) -> ArrayLike:
            initializer = (
                default_initializer if self._initializer is None else self._initializer
            )
            if callable(initializer):
                if shape is None or dtype is None:
                    raise ValueError(
                        "Shape and dtype must be set if initializer is callable."
                    )
                return initializer(shape, dtype)
            return self._initializer

        value = initialize(self._shape, self._dtype)
        try:
            return np.array(value)
        except Exception as e:
            raise ValueError(f"Invalid initializer: {e}")

    def __repr__(self) -> str:
        return f"Parameter(name={self._name}, shape={self._shape}, dtype={self._dtype})"


class SymbolicTensor(SupportsSpoxVar):
    def __init__(
        self,
        spox_var: Optional[spox.Var] = None,
        shape: Optional[ShapeLike] = None,
        dtype: Optional[DTypeLike] = None,
        source_layer: Optional[str] = None,
    ) -> None:
        if not isinstance(spox_var, spox.Var):
            if shape is None or dtype is None:
                raise ValueError("Shape and dtype must be set if spox_var is None.")
            self._spox_var = spox.argument(spox.Tensor(dtype=dtype, shape=shape))
        else:
            self._spox_var = spox_var
        self._shape = self._spox_var.unwrap_tensor().shape
        self._dtype = self._spox_var.unwrap_tensor().dtype
        self._source_layer = source_layer

    @property
    def shape(self) -> ShapeLike:
        return self._shape

    @property
    def dtype(self) -> DTypeLike:
        return self._dtype

    @property
    def source_layer(self) -> Optional[str]:
        return self._source_layer

    def spox_var(self) -> spox.Var:
        return self._spox_var

    def __repr__(self) -> str:
        out = f"SymbolicTensor(shape={self._shape}, dtype={self._dtype})"
        if self._source_layer is not None:
            out += f" from layer '{self._source_layer}'"
        return out


class EagerTensor(SymbolicTensor):
    def __init__(self, spox_var):
        super().__init__(spox_var=spox_var)

    @property
    def value(self) -> ndonnx.Array:
        return ndonnx.from_spox_var(self.spox_var())

    def __repr__(self) -> str:
        return f"EagerTensor(shape={self.shape}, dtype={self.dtype}): {self.value}"
