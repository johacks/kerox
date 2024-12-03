from typing import Callable

import keras
import spox
from keras import activations, saving

from kerox.core import KeroxTensor, in_onnx_build_scope
from kerox.ops import kops, sops, to_spox_var


def spox_constant_like(spox_var: spox.Var, value):
    tensor = spox_var.unwrap_tensor()
    return sops.const(value, dtype=tensor.dtype)


@saving.register_keras_serializable(package="kerox")
def relu(x):
    if in_onnx_build_scope():
        return KeroxTensor(spox_var=sops.relu(to_spox_var(x)))
    return kops.relu(x)


@saving.register_keras_serializable(package="kerox")
def sigmoid(x):
    if in_onnx_build_scope():
        return KeroxTensor(spox_var=sops.sigmoid(to_spox_var(x)))
    return kops.sigmoid(x)


@saving.register_keras_serializable(package="kerox")
def tanh(x):
    if in_onnx_build_scope():
        return KeroxTensor(spox_var=sops.tanh(to_spox_var(x)))
    return kops.tanh(x)


@saving.register_keras_serializable(package="kerox")
def softmax(x, axis=-1):
    if in_onnx_build_scope():
        return KeroxTensor(spox_var=sops.softmax(to_spox_var(x), axis=axis))
    return kops.softmax(x, axis=axis)


@saving.register_keras_serializable(package="kerox")
def relu6(x):
    if in_onnx_build_scope():
        x = to_spox_var(x)
        zero, six = spox_constant_like(x, 0), spox_constant_like(x, 6)
        return KeroxTensor(spox_var=sops.clip(x, zero, six))
    return kops.relu6(x)


@saving.register_keras_serializable(package="kerox")
def leaky_relu(x, negative_slope=0.3):
    if in_onnx_build_scope():
        x = to_spox_var(x)
        alpha = spox_constant_like(x, negative_slope)
        return KeroxTensor(spox_var=sops.leaky_relu(x, alpha=alpha))
    return kops.leaky_relu(x, negative_slope=negative_slope)


@saving.register_keras_serializable(package="kerox")
def elu(x, alpha=1.0):
    if in_onnx_build_scope():
        x = to_spox_var(x)
        alpha = spox_constant_like(x, alpha)
        return KeroxTensor(spox_var=sops.elu(x, alpha=alpha))
    return kops.elu(x, alpha=alpha)


@saving.register_keras_serializable(package="kerox")
def silu(x):
    if in_onnx_build_scope():
        x = to_spox_var(x)
        return KeroxTensor(spox_var=sops.mul(x, sops.sigmoid(x)))
    return kops.silu(x)


@saving.register_keras_serializable(package="kerox")
def hard_silu(x):
    if in_onnx_build_scope():
        x = to_spox_var(x)
        return KeroxTensor(spox_var=sops.hard_swish(x))
    return kops.hard_silu(x)


@saving.register_keras_serializable(package="kerox")
def swish(x):
    return silu(x)


@saving.register_keras_serializable(package="kerox")
def hard_swish(x):
    return hard_silu(x)


@saving.register_keras_serializable(package="kerox")
def celu(x, alpha=1.0):
    if in_onnx_build_scope():
        x = to_spox_var(x)
        return KeroxTensor(spox_var=sops.celu(x, alpha=spox_constant_like(x, alpha)))
    return kops.celu(x, alpha=alpha)


@saving.register_keras_serializable(package="kerox")
def soft_shrink(x, threshold=0.5):
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
def selu(x):
    if in_onnx_build_scope():
        return KeroxTensor(spox_var=sops.selu(to_spox_var(x)))
    return kops.selu(x)


@saving.register_keras_serializable(package="kerox")
def softplus(x):
    if in_onnx_build_scope():
        return KeroxTensor(spox_var=sops.softplus(to_spox_var(x)))
    return kops.softplus(x)


@saving.register_keras_serializable(package="kerox")
def softsign(x):
    if in_onnx_build_scope():
        return KeroxTensor(spox_var=sops.softsign(to_spox_var(x)))
    return kops.softsign(x)


@saving.register_keras_serializable(package="kerox")
def squareplus(x, b=4):
    if in_onnx_build_scope():
        x = to_spox_var(x)
        b = spox_constant_like(x, b)
        num = sops.add(x, sops.sqrt(sops.add(sops.square(x), b)))
        den = spox_constant_like(num, 2)
        return KeroxTensor(spox_var=sops.div(num, den))
    return kops.squareplus(x)


@saving.register_keras_serializable(package="kerox")
def gelu(x):
    if in_onnx_build_scope():
        return KeroxTensor(spox_var=sops.gelu(to_spox_var(x)))
    return kops.gelu(x)


@saving.register_keras_serializable(package="kerox")
def glu(x):
    if in_onnx_build_scope():
        return KeroxTensor(spox_var=sops.glu(to_spox_var(x)))
    return kops.glu(x)


@saving.register_keras_serializable(package="kerox")
def tanh_shrink(x):
    if in_onnx_build_scope():
        return KeroxTensor(spox_var=sops.mul(to_spox_var(x), sops.tanh(to_spox_var(x))))
    return kops.tanh_shrink(x)


@saving.register_keras_serializable(package="kerox")
def exponential(x):
    if in_onnx_build_scope():
        return KeroxTensor(spox_var=sops.exp(to_spox_var(x)))
    return kops.exp(x)


@saving.register_keras_serializable(package="kerox")
def hard_sigmoid(x):
    if in_onnx_build_scope():
        return KeroxTensor(spox_var=sops.hard_sigmoid(to_spox_var(x)))
    return kops.hard_sigmoid(x)


@saving.register_keras_serializable(package="kerox")
def hard_tanh(x):
    if in_onnx_build_scope():
        x = to_spox_var(x)
        minus_one, one = spox_constant_like(x, -1), spox_constant_like(x, 1)
        return KeroxTensor(spox_var=sops.clip(x, minus_one, one))
    return kops.hard_tanh(x)


@saving.register_keras_serializable(package="kerox")
def hard_shrink(x, threshold=0.5):
    if in_onnx_build_scope():
        x = to_spox_var(x)
        threshold = spox_constant_like(x, threshold)
        zero = spox_constant_like(x, 0)
        cond_x = sops.greater(sops.abs(x), threshold)
        return KeroxTensor(spox_var=sops.if_(cond_x, (x,), (zero,))[0])

    return kops.hard_shrink(x, threshold=threshold)


@saving.register_keras_serializable(package="kerox")
def linear(x):
    if in_onnx_build_scope():
        return KeroxTensor(spox_var=sops.identity(to_spox_var(x)))
    return x


@saving.register_keras_serializable(package="kerox")
def mish(x):
    if in_onnx_build_scope():
        return KeroxTensor(spox_var=sops.mish(to_spox_var(x)))
    return keras.activations.mish(x)


@saving.register_keras_serializable(package="kerox")
def log_softmax(x, axis=-1):
    if in_onnx_build_scope():
        return KeroxTensor(spox_var=sops.log_softmax(to_spox_var(x), axis=axis))
    return kops.log_softmax(x, axis=axis)


@saving.register_keras_serializable(package="kerox")
def log_sigmoid(x):
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
