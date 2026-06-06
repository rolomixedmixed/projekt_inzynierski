import cupy as cp
import cuquantum
from cuquantum import custatevec

n_qubits = 3
sv = cp.array([0,0,0,0,0,1,0,0], dtype=cp.complex128)

control_bit=[2]
target_bit=[0,1]
control_value=[1]

swap_gate=cp.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=cp.complex128)

handle=custatevec.create()

custatevec.apply_matrix(
    handle,
    sv.data.ptr,
    cuquantum.cudaDataType.CUDA_C_64F,
    n_qubits,
    swap_gate.data.ptr,
    cuquantum.cudaDataType.CUDA_C_64F,
    1,
    0,
    target_bit,
    len(target_bit),
    control_bit,
    control_value,
    len(control_bit),
    cuquantum.ComputeType.COMPUTE_64F,
    0,
    0
)

custatevec.destroy(handle)
print(sv)