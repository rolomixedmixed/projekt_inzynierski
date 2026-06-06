import cupy as cp
import cuquantum
from cuquantum import custatevec

n_qubits = 1

stan_gpu = cp.zeros(2**n_qubits, dtype=cp.complex128)
stan_gpu[0] = 1.0 + 0.0j 

bramka_x_gpu = cp.array([[0, 1], [1, 0]], dtype=cp.complex128)

handle = custatevec.create()

custatevec.apply_matrix(
    handle,                                  
    stan_gpu.data.ptr,                     
    cuquantum.cudaDataType.CUDA_C_64F,       
    n_qubits,                                
    bramka_x_gpu.data.ptr,                   
    cuquantum.cudaDataType.CUDA_C_64F,       
    custatevec.MatrixLayout.ROW,             
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

print(stan_gpu)

custatevec.destroy(handle)
