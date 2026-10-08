from dalia import xp


def ensure_scalar(value: float | xp.ndarray) -> float:
    """ Ensure that the input value is a scalar float. If the input is a 1-element array, extract the scalar value. 
    If the input is an array with more than 1 element, raise an error.
    
    Parameters
    ----------
    value : float or xp.ndarray
        The value to ensure is a scalar float.

    Returns
    -------
    float
        The scalar float value.

    Raises
    ------
    ValueError
        If the input is not a float or a 1-element array.

    """
    if isinstance(value, float):
        return value

    if isinstance(value, xp.ndarray):
        # Need to handle the case where the array is 0-dimensional 
        # (i.e., a scalar wrapped in an array)
        if value.ndim == 0:
            return float(value.item())

        if value.size > 1:
            raise ValueError(
                f"value evaluation returned an array of size {value.size}, expected a scalar."
            )
        return float(value[0])

    raise ValueError(
        f"value evaluation returned an object of type {type(value)}, expected a float or a 1-element array."
    )
