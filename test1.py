import cupy as cp
import cuquantum
from cuquantum import custatevec
import numpy as np
import math


n_qubits = 3

H = cp.asarray(
    [[1, 1],
     [1, -1]],
    dtype=cp.complex64
) / cp.sqrt(cp.float32(2.0))

X = cp.asarray(
    [[0, 1],
     [1, 0]],
    dtype=cp.complex64
)

Z = cp.asarray(
    [[1, 0],
     [0, -1]],
    dtype=cp.complex64
)

def alloc_workspace(handle, n_qubits, matrix, n_target, n_control):
    workspace_size=custatevec.apply_matrix_get_workspace_size(
        handle,
        cuquantum.cudaDataType.CUDA_C_32F,
        n_qubits,
        matrix.data.ptr,
        cuquantum.cudaDataType.CUDA_C_32F,
        custatevec.MatrixLayout.ROW,
        0,
        n_target,
        n_control,
        cuquantum.ComputeType.COMPUTE_32F,
    )

    workspace=None
    workspace_ptr=0
    if workspace_size>0:
        workspace=cp.cuda.alloc(workspace_size)
        workspace_ptr=workspace.ptr

    return workspace_size, workspace_ptr, workspace
    

def apply_gate(handle, sv_ptr, n_qubits, matrix, target, control=[], workspace_size=0, workspace_ptr=0):
    control_value = [1] * len(control)

    cuquantum.custatevec.apply_matrix(
        handle,
        sv_ptr,
        cuquantum.cudaDataType.CUDA_C_32F,
        n_qubits,
        matrix.data.ptr,
        cuquantum.cudaDataType.CUDA_C_32F,
        custatevec.MatrixLayout.ROW,
        0,
        target,
        len(target),
        control,
        control_value,
        len(control),
        cuquantum.ComputeType.COMPUTE_32F,
        workspace_ptr,
        workspace_size
    )


state = cp.zeros(2**n_qubits, dtype=cp.complex64)
state[0] = 1.0 + 0.0j

handle = custatevec.create()

[workspace_size, workspace_ptr, workspace]=alloc_workspace(handle, n_qubits, H, 1, 0)


for q in range(n_qubits):
    apply_gate(handle, state.data.ptr, n_qubits, H, [q], [], workspace_size, workspace_ptr)

cp.cuda.runtime.deviceSynchronize()

print("H:")
print(state.get())

del workspace
cp.get_default_memory_pool().free_all_blocks()

k=math.floor((np.pi/4.0)*np.sqrt(2**n_qubits))

for i in range(k):
    # ORACLE for |101> 
    #X
    [workspace_size, workspace_ptr, workspace]=alloc_workspace(handle, n_qubits, X, 1, 0)

    apply_gate(handle, state.data.ptr, n_qubits, X, [1], [], workspace_size, workspace_ptr)

    # CCZ
    [workspace_size, workspace_ptr, workspace]=alloc_workspace(handle, n_qubits, Z, 1, 2)

    apply_gate(handle, state.data.ptr, n_qubits, Z, [2], [0, 1], workspace_size, workspace_ptr)

    #X
    [workspace_size, workspace_ptr, workspace]=alloc_workspace(handle, n_qubits, X, 1, 0)

    apply_gate(handle, state.data.ptr, n_qubits, X, [1], [], workspace_size, workspace_ptr)

    cp.cuda.runtime.deviceSynchronize()

    #Diffusion operator:

    #H^n
    [workspace_size, workspace_ptr, workspace]=alloc_workspace(handle, n_qubits, H, 1, 0)

    for q in range(n_qubits):
        apply_gate(handle, state.data.ptr, n_qubits, H, [q], [], workspace_size, workspace_ptr)

    #X^n
    [workspace_size, workspace_ptr, workspace]=alloc_workspace(handle, n_qubits, X, 1, 0)
    for q in range(n_qubits):
        apply_gate(handle, state.data.ptr, n_qubits, X, [q], [], workspace_size, workspace_ptr)

    #CCZ
    [workspace_size, workspace_ptr, workspace]=alloc_workspace(handle, n_qubits, Z, 1, 2)

    apply_gate(handle, state.data.ptr, n_qubits, Z, [2], [0, 1], workspace_size, workspace_ptr)

    #X^n
    [workspace_size, workspace_ptr, workspace]=alloc_workspace(handle, n_qubits, X, 1, 0)
    for q in range(n_qubits):
        apply_gate(handle, state.data.ptr, n_qubits, X, [q], [], workspace_size, workspace_ptr)

    #H^n
    [workspace_size, workspace_ptr, workspace]=alloc_workspace(handle, n_qubits, H, 1, 0)

    for q in range(n_qubits):
        apply_gate(handle, state.data.ptr, n_qubits, H, [q], [], workspace_size, workspace_ptr)

print("results:")
print(state.get())

del workspace
cp.get_default_memory_pool().free_all_blocks() 

custatevec.destroy(handle)