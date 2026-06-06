import cupy as cp
import cuquantum
from cuquantum import custatevec
import numpy as np
import math
import random
from fractions import Fraction

def apply_gate(handle, sv_ptr, n_qubits, matrix, target, control=[]):
    control_value=[1]*len(control)

    workspace_size=custatevec.apply_matrix_get_workspace_size(
        handle,
        cuquantum.cudaDataType.CUDA_C_64F,
        n_qubits,
        matrix.data.ptr,
        cuquantum.cudaDataType.CUDA_C_64F,
        1,
        0,
        len(target),
        len(control),
        cuquantum.ComputeType.COMPUTE_64F,
    )

    workspace=None
    workspace_ptr=0
    if workspace_size>0:
        workspace=cp.cuda.alloc(workspace_size)
        workspace_ptr=workspace.ptr


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
        workspace_ptr,
        workspace_size
    )

h_gate=cp.array([[1, 1],
                 [1, -1]], dtype=cp.complex128)/cp.sqrt(2)

def r_gate(k, inv):
    return cp.array([[1, 0],
                     [0, np.exp(inv*1j*2*np.pi/2**k)]], dtype=cp.complex128)

def u_gate(x, power, N, L):
    matrix=cp.zeros((2**L,2**L), dtype=cp.complex128)
    multiplier=pow(int(x), int(2**power), int(N))

    for y in range(2**L):
        if y<N:
            matrix[((y*multiplier)%N),y]=1.0
        else:
            matrix[y,y]=1.0
    return matrix

def iqft(handle, sv_ptr, n_qubits, start_idx, end_idx):
    for target in range (start_idx, end_idx+1):
        for control in range(start_idx,target):
            k=target+1-control
            apply_gate(handle, sv_ptr, n_qubits, r_gate(k,-1), [target], [control])
        apply_gate(handle, sv_ptr, n_qubits, h_gate, [target])

def phase_estimation(handle, sv_ptr, t_qubits, L_qubits, x, N):
    n_qubits=t_qubits+L_qubits
    t_register=list(range(t_qubits, n_qubits))

    for idx in range(t_qubits):
        apply_gate(handle, sv_ptr, n_qubits, h_gate, [idx])
    for idx in range(t_qubits):
        u_matrix=u_gate(x, idx, N, L_qubits)
        apply_gate(handle, sv_ptr, n_qubits, u_matrix, t_register, [idx])
    iqft(handle, sv_ptr, n_qubits, 0, t_qubits-1)

def shors_algorithm(N):
    x=random.randint(2,N-1)
    if math.gcd(x,N)>1:
        print(f"znaleziono czynniki faktoryzacji {math.gcd(x,N)} i {N//math.gcd(x,N)}")
        return math.gcd(x,N), N//math.gcd(x,N)
    
    L_qubits=math.ceil(math.log2(N))
    t_qubits=L_qubits
    n_qubits=t_qubits+L_qubits

    sv=cp.zeros(2**n_qubits, dtype=cp.complex128)
    sv[1<<t_qubits]=1.0

    handle=custatevec.create()

    phase_estimation(handle, sv.data.ptr, t_qubits, L_qubits, x, N)
    probability=cp.asnumpy(cp.abs(sv)**2)

    custatevec.destroy(handle)

    reg1_prob=np.zeros(2**t_qubits)
    for i in range(len(probability)):
        reg1_prob[i%(2**t_qubits)]+=probability[i]
    
    measured_value = np.argmax(reg1_prob)

    if measured_value==0:
        print("measured 0 value")
        return None

    phase=measured_value/(2**t_qubits)
    fraction = fraction(phase).limit_denominator(N)
    r=fraction.denominator

    if pow(x,r,N) != 1:
        print("wrong order")
        return None
    
    if r%2 !=0:
       print("r is uneven number") 
       return None
    
    factor1=math.gcd(pow(x,r//2)-1,N)
    factor2=math.gcd(pow(x,r//2)+1,N)

    return factor1,factor2

if __name__=='__main__':
    N=1271
    factors=None
    attempts=0

    while factors is None and attempts<10:
        attempts+=1
        factors=shors_algorithm(N)
    if factors:
        print(f"factorization succesful, {factors[0]} and {factors[1]}")
    else:
        print("number of attempts exceeded")
