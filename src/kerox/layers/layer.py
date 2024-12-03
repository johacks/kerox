from typing import Optional

from keras import constraints, initializers, regularizers, utils
from keras import layers as klayers
from keras.src import backend
from optree import PyTree

from kerox.core import KeroxTensor, KeroxVariable, ONNXBuildScope, in_onnx_build_scope
from kerox.typing import ArrayOrTensor, ShapeLike


class Layer(klayers.Layer):
    def add_weight(
        self,
        shape: Optional[ShapeLike] = None,
        initializer=None,
        dtype=None,
        trainable=True,
        autocast=True,
        regularizer=None,
        constraint=None,
        aggregation="mean",
        name=None,
    ):
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
        self._check_super_called()
        if shape is None:
            shape = ()
        if dtype is not None:
            dtype = utils.standardize_dtype(dtype)
        else:
            dtype = self.variable_dtype
        if initializer is None:
            if "float" in dtype:
                initializer = "glorot_uniform"
            else:
                initializer = "zeros"
        initializer = initializers.get(initializer)
        with backend.name_scope(self.name, caller=self):
            variable = KeroxVariable(
                initializer=initializer,
                shape=shape,
                dtype=dtype,
                trainable=trainable,
                autocast=autocast,
                aggregation=aggregation,
                name=name,
            )
        # Will be added to layer.losses
        variable.regularizer = regularizers.get(regularizer)
        variable.constraint = constraints.get(constraint)
        self._track_variable(variable)
        return variable

    def __call__(self, *args, **kwargs) -> PyTree[ArrayOrTensor]:
        # Check if we are building in ONNX scope
        all_symbolic = all(isinstance(arg, KeroxTensor) for arg in args)
        any_symbolic = any(isinstance(arg, KeroxTensor) for arg in args)
        if not all_symbolic and any_symbolic:
            raise ValueError("All inputs must be KeroxTensor, or none of them.")
        if all_symbolic:
            if not in_onnx_build_scope():
                raise ValueError("KeroxTensor can only be used in ONNX build scope.")
            return self.call(*args, **kwargs)
        return super().__call__(*args, **kwargs)

    def onnx_symbolic_call(self, *args: KeroxTensor, **kwargs) -> PyTree[KeroxTensor]:
        with ONNXBuildScope():
            return self(*args, **kwargs)
