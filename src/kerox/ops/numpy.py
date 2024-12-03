from typing import Sequence, TypeAlias

import ndonnx
import numpy as np
import spox
import spox.opset.ai.onnx.v21 as sops
from jaxtyping import ArrayLike
from keras import ops as kops

from kerox import core

ArrayOrTensor: TypeAlias = core.Variable | ArrayLike | spox.Var


def to_spox_var(x: ArrayOrTensor) -> spox.Var:
    if isinstance(x, (core.Variable, core.KeroxTensor, ndonnx.Array)):
        return x.spox_var()
    if isinstance(x, spox.Var):
        return x
    x = np.array(x)
    return spox.argument(spox.Tensor(x.dtype, x.shape))


def many_to_spox_var(*xs: ArrayOrTensor) -> Sequence[spox.Var]:
    return tuple(to_spox_var(x) for x in xs)


def matmul(x1: ArrayOrTensor, x2: ArrayOrTensor) -> ArrayOrTensor:
    if core.in_onnx_build_scope():
        x1, x2 = many_to_spox_var(x1, x2)
        return core.KeroxTensor(spox_var=sops.matmul(x1, x2))
    return kops.matmul(x1, x2)


def add(x1: ArrayOrTensor, x2: ArrayOrTensor) -> ArrayOrTensor:
    if core.in_onnx_build_scope():
        x1, x2 = many_to_spox_var(x1, x2)
        return core.KeroxTensor(spox_var=sops.add(x1, x2))
    return kops.add(x1, x2)
