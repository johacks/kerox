from typing import Callable, Sequence, SupportsIndex, TypeAlias

import spox
import spox._future
from numpy.typing import ArrayLike, DTypeLike

from kerox import core

ShapeLike: TypeAlias = SupportsIndex | Sequence[SupportsIndex]
Activation: TypeAlias = Callable[[spox.Var], spox.Var]
ArrayOrTensor: TypeAlias = (
    "core.KeroxVariable" | ArrayLike | spox.Var | "core.KeroxTensor"
)
