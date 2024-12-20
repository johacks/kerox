from functools import wraps
from typing import Callable, Sequence

import ndonnx
import numpy as np
import spox
from keras import ops as kops  # noqa: F401
from spox.opset.ai.onnx import v21 as sops  # noqa: F401

from kerox import core
from kerox.typing import ArrayOrTensor


def to_spox_var(x: ArrayOrTensor) -> spox.Var:
    if isinstance(x, (core.KeroxVariable, core.KeroxTensor, ndonnx.Array)):
        return x.spox_var()
    if isinstance(x, spox.Var):
        return x
    x = np.array(x)
    return spox.argument(spox.Tensor(x.dtype, x.shape))


def many_to_spox_var(*xs: ArrayOrTensor) -> Sequence[spox.Var]:
    return tuple(to_spox_var(x) for x in xs)


def spox_auto_adapt_op(keras_func: Callable, spox_func: Callable) -> Callable:
    def inner_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if core.in_onnx_build_scope():
                args = many_to_spox_var(*args)
                return core.KeroxTensor(spox_var=spox_func(*args, **kwargs))
            return keras_func(*args, **kwargs)

        return wrapper

    return inner_wrapper


def spox_constant_like(spox_var: spox.Var, value):
    tensor = spox_var.unwrap_tensor()
    return sops.const(value, dtype=tensor.dtype)
