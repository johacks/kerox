from typing import TYPE_CHECKING, Optional

import spox
import spox._future
from keras import KerasTensor
from keras.src.backend.common import global_state

from kerox.ops.utils import sops
from kerox.typing import DTypeLike, ShapeLike

if TYPE_CHECKING:
    from keras.src.backend.tensorflow import Variable as KerasVariable
else:
    from keras import Variable as KerasVariable


class ONNXBuildScope:
    def __enter__(self):
        self._already_in_onnx_build = in_onnx_build_scope()
        global_state.set_global_attribute("onnx_build", True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self._already_in_onnx_build:
            global_state.set_global_attribute("onnx_build", None)


def in_onnx_build_scope() -> bool:
    return global_state.get_global_attribute("onnx_build", default=None) is not None


class KeroxVariable(KerasVariable):
    def spox_var(self) -> spox.Var:
        if self.trainable:
            # Allows training in onnxruntime for training
            var = spox._future.initializer(value=self.numpy())
        else:
            # Don't risk using experimental feature if we are sure it's not trainable
            var = sops.constant(value=self.numpy())
        var._rename(self.path)
        return var

    def __repr__(self):
        return KerasVariable.__repr__(self).replace("Variable", "KeroxVariable")


class KeroxTensor(KerasTensor):
    def __init__(
        self,
        shape: Optional[ShapeLike] = None,
        dtype: DTypeLike = "float32",
        spox_var: Optional[spox.Var] = None,
        name: Optional[str] = None,
        **kwargs,
    ):
        if spox_var is not None:
            tensor = spox_var.unwrap_tensor()
            shape, dtype = tensor.shape, tensor.dtype
        else:
            if shape is None:
                raise ValueError("shape must be provided if spox_var is not.")
        self._spox_var = spox_var
        super().__init__(shape, dtype, name=name, **kwargs)

    def spox_var(self) -> spox.Var:
        if self._spox_var is not None:
            return self._spox_var
        var = spox.argument(spox.Tensor(self.dtype, self.shape))
        var._rename(self.name)
        self._spox_var = var
        return var

    def __repr__(self):
        return "<KeroxTensor: shape={}, dtype={}, name={}>".format(
            self.shape, self.dtype, self.name
        )
