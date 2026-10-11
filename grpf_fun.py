import numpy as np

def fun(z):
    """
    Compute the function value based on the given argument.

    Args:
    z (complex): Function argument.

    Returns:
    complex: Function value.

    adapted from: https://github.com/PG-Group-Project-Quantum-Optimization/GRPF-quantum.git
    """
    try:
            w = (z-1)*(z-1j)**2*(z+1)**3/(z+1j)
    except:
        w=complex(np.inf, np.inf)
    return w