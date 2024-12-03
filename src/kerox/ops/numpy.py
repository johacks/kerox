from kerox.ops.utils import direct_conversion, kops, sops
from kerox.typing import ArrayOrTensor


@direct_conversion(kops.matmul, sops.matmul)
def matmul(x1: ArrayOrTensor, x2: ArrayOrTensor) -> ArrayOrTensor: ...


@direct_conversion(kops.add, sops.add)
def add(x1: ArrayOrTensor, x2: ArrayOrTensor) -> ArrayOrTensor: ...
