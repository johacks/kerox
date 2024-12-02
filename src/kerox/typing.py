from typing import Callable, Protocol, Sequence, SupportsIndex, TypeAlias

import spox
import spox._future
from numpy.typing import ArrayLike, DTypeLike

ShapeLike: TypeAlias = SupportsIndex | Sequence[SupportsIndex]
Initializer: TypeAlias = Callable[[ShapeLike, DTypeLike], ArrayLike] | ArrayLike
Activation: TypeAlias = Callable[[spox.Var], spox.Var]


class SupportsSpoxVar(Protocol):
    def spox_var(self) -> spox.Var: ...
