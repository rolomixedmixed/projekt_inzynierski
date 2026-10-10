def vinq(f):
    """
    Converts the function values to quadrants.

    Args:
        f (float): Function value.

    Returns:
        int: Quadrant.

    Based on: https://github.com/PG-Group-Project-Quantum-Optimization/GRPF-quantum.git
    """

    if f.real > 0 and f.imag >= 0:
        w = 0
    elif f.real <= 0 and f.imag > 0:
        w = 1
    elif f.real < 0 and f.imag <= 0:
        w = 2
    elif f.real >= 0 and f.imag < 0:
        w = 3
    else:
        w = None

    return w