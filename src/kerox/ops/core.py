from keras import utils

from kerox import core, ops


def convert_to_tensor(x, dtype=None, sparse=None):
    """Convert a NumPy array to a tensor.

    Args:
        x: A NumPy array, Python array (can be nested) or a backend tensor.
        dtype: The target type. If `None`, the type of `x` is used.
        sparse: Whether to keep sparse tensors. `False` will cause sparse
            tensors to be densified. The default value of `None` means that
            sparse tensors are kept only if the backend supports them.

    Returns:
        A backend tensor of the specified `dtype` and sparseness.

    Example:

    >>> x = np.array([1, 2, 3])
    >>> y = keras.ops.convert_to_tensor(x)
    """
    if isinstance(x, core.KeroxTensor):
        if bool(sparse):
            raise ValueError(
                f"When passing a KeroxTensor, sparse should not be set. Got {sparse}"
            )
        inner_var = ops.to_spox_var(x)
        if dtype is None:
            return x
        if utils.standardize_dtype(dtype) == utils.standardize_dtype(
            inner_var.unwrap_tensor().dtype
        ):
            return x
        return ops.cast(inner_var, dtype=dtype)
    return ops.kops.convert_to_tensor(x, dtype=dtype, sparse=sparse)
