import numpy as np
from vinq import vinq

def evaluate_quadrants(nodes, func):
    num_nodes = len(nodes)
    quadrants = np.zeros(num_nodes, dtype=np.int32)

    for i in range(num_nodes):
        z=complex(nodes[i][0], nodes[i][1])
        q_val=vinq(func(z))

        quadrants[i]=q_val if q_val is not None else 0

    return quadrants

