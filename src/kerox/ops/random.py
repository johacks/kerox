import numpy as np
from keras.src.backend import random as krandom
from keras.src.random.seed_generator import draw_seed

from kerox.core import KeroxTensor, in_onnx_build_scope
from kerox.ops.utils import kops, sops, spox_constant_like, to_spox_var
from kerox.typing import ArrayOrTensor


def as_int_seed(seed):
    a, b = kops.convert_to_numpy(draw_seed(seed))
    return int(a + b * 2**32)


def dropout(inputs: ArrayOrTensor, rate: float, seed=None) -> ArrayOrTensor:
    if in_onnx_build_scope():
        seed = as_int_seed(seed)
        inputs = to_spox_var(inputs)
        rate = spox_constant_like(inputs, rate)
        train = sops.constant(value=np.array(True))
        return KeroxTensor(spox_var=sops.dropout(inputs, rate, train, seed=seed)[0])
    return krandom.dropout(inputs, rate=rate, seed=seed)
