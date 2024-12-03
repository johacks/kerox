from abc import ABC
from functools import wraps

from keras import Operation
from keras import layers as klayers
from optree import PyTree

from kerox.core import KeroxTensor, KeroxVariable, ONNXBuildScope, in_onnx_build_scope


class Layer(klayers.Layer, ABC):
    @wraps(klayers.Layer.add_weight)
    def add_weight(self, *args, **kwargs):
        """Add a weight variable to the layer.

        Args:
            shape: Shape tuple for the variable. Must be fully-defined
                (no `None` entries). Defaults to `()` (scalar) if unspecified.
            initializer: Initializer object to use to populate the initial
                variable value, or string name of a built-in initializer
                (e.g. `"random_normal"`). If unspecified, defaults to
                `"glorot_uniform"` for floating-point variables and to `"zeros"`
                for all other types (e.g. int, bool).
            dtype: Dtype of the variable to create, e.g. `"float32"`. If
                unspecified, defaults to the layer's variable dtype
                (which itself defaults to `"float32"` if unspecified).
            trainable: Boolean, whether the variable should be trainable via
                backprop or whether its updates are managed manually. Defaults
                to `True`.
            autocast: Boolean, whether to autocast layers variables when
                accessing them. Defaults to `True`.
            regularizer: Regularizer object to call to apply penalty on the
                weight. These penalties are summed into the loss function
                during optimization. Defaults to `None`.
            constraint: Contrainst object to call on the variable after any
                optimizer update, or string name of a built-in constraint.
                Defaults to `None`.
            aggregation: String, one of `'mean'`, `'sum'`,
                `'only_first_replica'`. Annotates the variable with the type
                of multi-replica aggregation to be used for this variable
                when writing custom data parallel training loops.
            name: String name of the variable. Useful for debugging purposes.
        """
        import keras.src.layers.layer as _parent_module

        # Patch the backend Variable to be KeroxVariable for the duration of the call
        Variable = _parent_module.backend.Variable
        _parent_module.backend.Variable = KeroxVariable
        try:
            result = super().add_weight(*args, **kwargs)
        finally:
            _parent_module.backend.Variable = Variable
        return result

    def symbolic_call(self, *args, **kwargs):
        # Whenever building the ONNX model, we want to call the layer's `call` method
        if in_onnx_build_scope():
            if all(isinstance(arg, KeroxTensor) for arg in args):
                return self.call(*args, **kwargs)
            else:
                raise ValueError(
                    f"Expected all arguments to be KeroxTensor when in ONNX build scope, but got {args}"
                )
        return Operation.symbolic_call(self, *args, **kwargs)

    def onnx_symbolic_call(self, *args: KeroxTensor, **kwargs) -> PyTree[KeroxTensor]:
        with ONNXBuildScope():
            return self(*args, **kwargs)
