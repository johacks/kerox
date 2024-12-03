from kerox.ops.utils import kops, sops, spox_auto_adapt_op
from kerox.typing import ArrayOrTensor


@spox_auto_adapt_op(kops.matmul, sops.matmul)
def matmul(x1: ArrayOrTensor, x2: ArrayOrTensor) -> ArrayOrTensor: ...


@spox_auto_adapt_op(kops.add, sops.add)
def add(x1: ArrayOrTensor, x2: ArrayOrTensor) -> ArrayOrTensor: ...
