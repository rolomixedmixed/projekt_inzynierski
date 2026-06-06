import cupy as cp
import cuquantum
from cuquantum import custatevec

state=cp.zeros(2, dtype=cp.complex128)
state[0]=1.0

n_qubits=1

print(state)

hadamard=cp.array([[1, 1],[1, -1]], dtype=cp.complex128)/cp.sqrt(2)

print(hadamard)

handle=custatevec.create()

custatevec.apply_matrix(
    handle,
    state.data.ptr,
    cuquantum.cudaDataType.CUDA_C_64F,
    n_qubits,
    hadamard.data.ptr,
    cuquantum.cudaDataType.CUDA_C_64F,
    1,    #custatevec.MatrixLayout.ROW,
    0,
    [0],
    1,
    [],
    [],
    0,
    cuquantum.ComputeType.COMPUTE_64F,
    0,
    0
    )

print(state)

custatevec.destroy(handle)
