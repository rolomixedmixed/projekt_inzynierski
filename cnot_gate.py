import cupy as cp
import cuquantum
from cuquantum import custatevec

sv=cp.asarray([0.0,0.0,1.0,0.0], dtype=cp.complex128)
n_qubits=2

x_gate=cp.array([[0, 1],[1, 0]], dtype=cp.complex128)

# |q1q0>, q1=control  q0=target
target_qubit=[0]
control_qubit=[1]
control_value=[1]

handle=custatevec.create()

custatevec.apply_matrix(
    handle,
    sv.data.ptr,
    cuquantum.cudaDataType.CUDA_C_64F,
    n_qubits,
    x_gate.data.ptr,
    cuquantum.cudaDataType.CUDA_C_64F,
    1,
    0,
    target_qubit,
    len(target_qubit),
    control_qubit,
    control_value,
    len(control_qubit),
    cuquantum.ComputeType.COMPUTE_64F,
    0,
    0
)

custatevec.destroy(handle)

print(sv)