import numpy as np

def fun(z):
    """
    Compute the function value based on the given argument.

    Args:
    z (complex): Function argument.

    Returns:
    complex: Function value.
    """
    try:
        w = (z-0.5)/((z+0.5)**2)
    except:
        w=complex(np.inf, np.inf)
    return w