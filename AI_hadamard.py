import cupy as cp

inv_sqrt2 = 1.0 / cp.sqrt(2.0)

hadamard_gate = cp.array([
    [inv_sqrt2,  inv_sqrt2],
    [inv_sqrt2, -inv_sqrt2]
], dtype=cp.complex128)

print(hadamard_gate)