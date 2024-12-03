from typing import Callable

from keras import activations, saving

from kerox.core import KeroxTensor, in_onnx_build_scope
from kerox.ops.utils import (
    kops,
    sops,
    spox_auto_adapt_op,
    spox_constant_like,
    to_spox_var,
)
from kerox.typing import ArrayOrTensor


@saving.register_keras_serializable(package="kerox")
@spox_auto_adapt_op(kops.relu, sops.relu)
def relu(x: ArrayOrTensor) -> ArrayOrTensor: ...


@saving.register_keras_serializable(package="kerox")
@spox_auto_adapt_op(kops.sigmoid, sops.sigmoid)
def sigmoid(x: ArrayOrTensor) -> ArrayOrTensor: ...


@saving.register_keras_serializable(package="kerox")
@spox_auto_adapt_op(kops.tanh, sops.tanh)
def tanh(x: ArrayOrTensor) -> ArrayOrTensor: ...


@saving.register_keras_serializable(package="kerox")
@spox_auto_adapt_op(kops.softmax, sops.softmax)
def softmax(x: ArrayOrTensor, *, axis=-1) -> ArrayOrTensor: ...


@saving.register_keras_serializable(package="kerox")
def relu6(x: ArrayOrTensor) -> ArrayOrTensor:
    if in_onnx_build_scope():
        x = to_spox_var(x)
        zero, six = spox_constant_like(x, 0), spox_constant_like(x, 6)
        return KeroxTensor(spox_var=sops.clip(x, zero, six))
    return kops.relu6(x)


@saving.register_keras_serializable(package="kerox")
def leaky_relu(x: ArrayOrTensor, *, negative_slope=0.3) -> ArrayOrTensor:
    if in_onnx_build_scope():
        x = to_spox_var(x)
        return KeroxTensor(spox_var=sops.leaky_relu(x, alpha=negative_slope))
    return kops.leaky_relu(x, negative_slope=negative_slope)


@saving.register_keras_serializable(package="kerox")
@spox_auto_adapt_op(kops.elu, sops.elu)
def elu(x: ArrayOrTensor, *, alpha=1.0) -> ArrayOrTensor: ...


@saving.register_keras_serializable(package="kerox")
def silu(x: ArrayOrTensor) -> ArrayOrTensor:
    if in_onnx_build_scope():
        x = to_spox_var(x)
        return KeroxTensor(spox_var=sops.mul(x, sops.sigmoid(x)))
    return kops.silu(x)


@saving.register_keras_serializable(package="kerox")
@spox_auto_adapt_op(kops.hard_silu, sops.hard_swish)
def hard_silu(x: ArrayOrTensor) -> ArrayOrTensor: ...


@saving.register_keras_serializable(package="kerox")
def swish(x: ArrayOrTensor) -> ArrayOrTensor:
    return silu(x)


@saving.register_keras_serializable(package="kerox")
def hard_swish(x: ArrayOrTensor) -> ArrayOrTensor:
    return hard_silu(x)


@saving.register_keras_serializable(package="kerox")
@spox_auto_adapt_op(kops.celu, sops.celu)
def celu(x: ArrayOrTensor, *, alpha=1.0) -> ArrayOrTensor: ...


@saving.register_keras_serializable(package="kerox")
def soft_shrink(x: ArrayOrTensor, threshold=0.5) -> ArrayOrTensor:
    if in_onnx_build_scope():
        x = to_spox_var(x)
        plus_threshold = spox_constant_like(x, threshold)
        minus_threshold = spox_constant_like(x, -threshold)
        zero = spox_constant_like(x, 0)
        result = sops.if_(
            sops.greater(x, plus_threshold),
            (sops.sub(x, plus_threshold),),
            sops.if_(
                sops.less(x, minus_threshold), (sops.add(x, plus_threshold),), (zero,)
            ),
        )[0]
        return KeroxTensor(spox_var=result)
    return kops.soft_shrink(x, threshold=threshold)


@saving.register_keras_serializable(package="kerox")
@spox_auto_adapt_op(kops.selu, sops.selu)
def selu(x: ArrayOrTensor) -> ArrayOrTensor: ...


@saving.register_keras_serializable(package="kerox")
@spox_auto_adapt_op(kops.softplus, sops.softplus)
def softplus(x: ArrayOrTensor) -> ArrayOrTensor: ...


@saving.register_keras_serializable(package="kerox")
@spox_auto_adapt_op(kops.softsign, sops.softsign)
def softsign(x: ArrayOrTensor) -> ArrayOrTensor: ...


@saving.register_keras_serializable(package="kerox")
def squareplus(x: ArrayOrTensor, b=4) -> ArrayOrTensor:
    if in_onnx_build_scope():
        x = to_spox_var(x)
        b = spox_constant_like(x, b)
        num = sops.add(x, sops.sqrt(sops.add(sops.square(x), b)))
        den = spox_constant_like(num, 2)
        return KeroxTensor(spox_var=sops.div(num, den))
    return kops.squareplus(x)


@saving.register_keras_serializable(package="kerox")
def gelu(x: ArrayOrTensor, *, approximate=True) -> ArrayOrTensor:
    if in_onnx_build_scope():
        x = to_spox_var(x)
        result = sops.gelu(x, approximate="tanh" if approximate else "none")
        return KeroxTensor(spox_var=result)
    return kops.gelu(x, approximate=approximate)


@saving.register_keras_serializable(package="kerox")
def glu(x: ArrayOrTensor, *, axis=-1) -> ArrayOrTensor:
    if in_onnx_build_scope():
        x = to_spox_var(x)
        a, b = sops.split(x, axis=axis, num_outputs=2)
        return KeroxTensor(spox_var=sops.mul(a, sops.sigmoid(b)))
    return kops.glu(x, axis=axis)


@saving.register_keras_serializable(package="kerox")
def tanh_shrink(x: ArrayOrTensor) -> ArrayOrTensor:
    if in_onnx_build_scope():
        return KeroxTensor(spox_var=sops.mul(to_spox_var(x), sops.tanh(to_spox_var(x))))
    return kops.tanh_shrink(x)


@saving.register_keras_serializable(package="kerox")
@spox_auto_adapt_op(kops.exp, sops.exp)
def exponential(x: ArrayOrTensor) -> ArrayOrTensor: ...


@saving.register_keras_serializable(package="kerox")
@spox_auto_adapt_op(kops.hard_sigmoid, sops.hard_sigmoid)
def hard_sigmoid(x: ArrayOrTensor) -> ArrayOrTensor: ...


@saving.register_keras_serializable(package="kerox")
def hard_tanh(x: ArrayOrTensor) -> ArrayOrTensor:
    if in_onnx_build_scope():
        x = to_spox_var(x)
        minus_one, one = spox_constant_like(x, -1), spox_constant_like(x, 1)
        return KeroxTensor(spox_var=sops.clip(x, minus_one, one))
    return kops.hard_tanh(x)


@saving.register_keras_serializable(package="kerox")
def hard_shrink(x: ArrayOrTensor, threshold=0.5) -> ArrayOrTensor:
    if in_onnx_build_scope():
        x = to_spox_var(x)
        threshold = spox_constant_like(x, threshold)
        zero = spox_constant_like(x, 0)
        cond_x = sops.greater(sops.abs(x), threshold)
        return KeroxTensor(spox_var=sops.if_(cond_x, (x,), (zero,))[0])

    return kops.hard_shrink(x, threshold=threshold)


@saving.register_keras_serializable(package="kerox")
def linear(x: ArrayOrTensor):
    if in_onnx_build_scope():
        return KeroxTensor(spox_var=sops.identity(to_spox_var(x)))
    return x


@saving.register_keras_serializable(package="kerox")
@spox_auto_adapt_op(activations.mish, sops.mish)
def mish(x: ArrayOrTensor) -> ArrayOrTensor: ...


@saving.register_keras_serializable(package="kerox")
@spox_auto_adapt_op(kops.log_softmax, sops.log_softmax)
def log_softmax(x: ArrayOrTensor, *, axis=-1) -> ArrayOrTensor: ...


@saving.register_keras_serializable(package="kerox")
def log_sigmoid(x: ArrayOrTensor) -> ArrayOrTensor:
    if in_onnx_build_scope():
        return KeroxTensor(spox_var=sops.log(sops.sigmoid(to_spox_var(x))))
    return kops.log_sigmoid(x)


ALL_OBJECTS = {
    relu,
    leaky_relu,
    relu6,
    softmax,
    celu,
    elu,
    selu,
    softplus,
    softsign,
    squareplus,
    soft_shrink,
    silu,
    gelu,
    glu,
    tanh,
    tanh_shrink,
    sigmoid,
    exponential,
    hard_sigmoid,
    hard_silu,
    hard_tanh,
    hard_shrink,
    linear,
    mish,
    log_softmax,
    log_sigmoid,
}
NAME_TO_FUNCTION = {obj.__name__: obj for obj in ALL_OBJECTS}
NAME_TO_FUNCTION[None] = linear


def get(name_or_callable: str | Callable):
    if callable(name_or_callable):
        return name_or_callable

    if name_or_callable not in NAME_TO_FUNCTION:
        raise ValueError(f"Unknown activation function: {name_or_callable}")
    return NAME_TO_FUNCTION[name_or_callable]


def serialize(function):
    return activations.serialize(function)


def deserialize(name, custom_objects=None):
    return activations.deserialize(name, custom_objects=custom_objects)
