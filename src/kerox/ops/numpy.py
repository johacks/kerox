from kerox import core
from kerox.ops.utils import kops, sops, spox_auto_adapt_op, to_spox_var
from kerox.typing import ArrayOrTensor


@spox_auto_adapt_op(kops.matmul, sops.matmul)
def matmul(x1: ArrayOrTensor, x2: ArrayOrTensor) -> ArrayOrTensor: ...


@spox_auto_adapt_op(kops.add, sops.add)
def add(x1: ArrayOrTensor, x2: ArrayOrTensor) -> ArrayOrTensor: ...


def cast(x: ArrayOrTensor, dtype: str) -> ArrayOrTensor:
    if core.in_onnx_build_scope():
        x = to_spox_var(x)
        new_var = sops.cast(x, to=dtype)
        return core.KeroxTensor(spox_var=new_var)
    return kops.cast(x, dtype=dtype)
