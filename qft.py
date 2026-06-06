import cupy as cp
import cuquantum
from cuquantum import custatevec
import numpy as np

def apply_gate(handle, sv_ptr, n_qubits, matrix, target, control=[]):
    control_value=[1]*len(control)

    custatevec.apply_matrix(
        handle,
        sv_ptr,
        cuquantum.cudaDataType.CUDA_C_64F,
        n_qubits,
        matrix.data.ptr,
        cuquantum.cudaDataType.CUDA_C_64F,
        1,
        0,
        target,
        len(target),
        control,
        control_value,
        len(control),
        cuquantum.ComputeType.COMPUTE_64F,
        0,
        0
    )

h_gate=cp.array([[1, 1],
                 [1, -1]], dtype=cp.complex128)/cp.sqrt(2)

def r_gate(k, inv):
    return cp.array([[1, 0],
                     [0, np.exp(inv*1j*2*np.pi/2**k)]], dtype=cp.complex128)

def u_gate(theta, power):
    return cp.array([[1, 0],
                     [0, np.exp(1j*2*np.pi*theta*(2**power))]], dtype=cp.complex128)


def qft(handle, sv_ptr, n_qubits):
    for target in range(n_qubits-1, -1, -1):
        apply_gate(handle, sv_ptr, n_qubits, h_gate, [target])        
        for control in range(target-1,-1,-1):
            k=target+1-control
            apply_gate(handle, sv_ptr, n_qubits, r_gate(k,1), [target], [control])

def iqft(handle, sv_ptr, n_qubits, start_idx=1):
    for target in range (start_idx, n_qubits):
        for control in range(start_idx,target):
            k=target+1-control
            apply_gate(handle, sv_ptr, n_qubits, r_gate(k,-1), [target], [control])
        apply_gate(handle, sv_ptr, n_qubits, h_gate, [target])

def phase_estimation(handle, sv_ptr, n_qubits, theta):
    for idx in range(1,n_qubits,1):
        apply_gate(handle, sv_ptr, n_qubits, h_gate, [idx])
        apply_gate(handle, sv_ptr, n_qubits, u_gate(theta, n_qubits-idx-1), [0], [idx])
    iqft(handle, sv_ptr, n_qubits)


if __name__=='__main__':
    n_qubits=5
    sv=cp.zeros(2**n_qubits, dtype=cp.complex128)
    sv[1]=1.0

    handle=custatevec.create()
    desired_phase=1/16
    phase_estimation(handle, sv.data.ptr, n_qubits, desired_phase)

    probability=cp.abs(sv)**2

    print(cp.round(probability, 4))
    custatevec.destroy(handle)


