import cupy as cp

if(False):
    state_vector=cp.zeros(2**3, dtype=cp.complex128)
    state_vector1=cp.ones(2**3, dtype=cp.complex64)
    state_vector[0]=1.0+1.0j
    print(state_vector1)

if(False):
    x_gate = [[0, 1],[1, 0]]
    x_gate_gpu = cp.array(x_gate, dtype=cp.complex128)
    n=cp.array([1,0], dtype=cp.complex64)
    mult=state*x_gate_gpu
    print(neg_state)
    print(cp.asnumpy(x_gate_gpu))

if(False):
    print(neg_state.data.ptr)


