import cupy as cp
import cuquantum
from cuquantum import custatevec
import numpy as np

class GroverAlgorithm:
    def __init__(self, n_qubits:int):
        if n_qubits<2:
            raise ValueError("not enough qubits")

        self.n_qubits=n_qubits
        self.dim=1<<n_qubits

        #state
        self.state=cp.zeros(self.dim, dtype=cp.complex64)
        self.state[0]=1.0+0.0j

        #gate matrices
        self.H=cp.asarray([[1.0, 1.0],
                           [1.0, -1.0]], dtype=cp.complex64)/cp.sqrt(cp.float32(2.0))
        self.X=cp.asarray([[0.0, 1.0],
                            [1.0, 0.0]], dtype=cp.complex64)
        self.Z=cp.asarray([[1.0, 0.0],
                            [0.0, -1.0]], dtype=cp.complex64)

        self.handle=custatevec.create()

        self.workspace_1qubit=self.alloc_workspace(self.H, 1, 0)

        self.workspace_mcz=self.alloc_workspace(self.Z, 1, self.n_qubits-1)


    def alloc_workspace(self, matrix, n_target :int, n_control :int):
        size=custatevec.apply_matrix_get_workspace_size(
            self.handle,
            cuquantum.cudaDataType.CUDA_C_32F,
            self.n_qubits,
            matrix.data.ptr,
            cuquantum.cudaDataType.CUDA_C_32F,
            custatevec.MatrixLayout.ROW,
            0,
            n_target,
            n_control,
            cuquantum.ComputeType.COMPUTE_32F,
        )
        if size>0:
            memory=cp.cuda.alloc(size)
            return memory, memory.ptr, size
        else:
            return None, 0, 0

    def apply_matrix(self, matrix, target, control, workspace):

        [memory, workspace_ptr, workspace_size]=workspace

        if control is None:
            control=0
            control_values=0
            n_control=0
        else:
            n_control=len(control)
            control_values=[1]*n_control

        cuquantum.custatevec.apply_matrix(
            self.handle,
            self.state.data.ptr,
            cuquantum.cudaDataType.CUDA_C_32F,
            self.n_qubits,
            matrix.data.ptr,
            cuquantum.cudaDataType.CUDA_C_32F,
            custatevec.MatrixLayout.ROW,
            0,
            target,
            len(target),
            control,
            control_values,
            n_control,
            cuquantum.ComputeType.COMPUTE_32F,
            workspace_ptr,
            workspace_size
        )

    def gate_1q(self, matrix, target):
        if target<0 or target >= self.n_qubits:
            raise ValueError("incorrect target qubit")
        self.apply_matrix(matrix, [target], None, self.workspace_1qubit)

    def gate_mcz(self):
        target=[self.n_qubits-1]
        control=list(range(self.n_qubits-1))
        self.apply_matrix(self.Z, target, control, self.workspace_mcz)

    def applyH_n(self):
        for q in range(self.n_qubits):
            self.gate_1q(self.H, q)

    def oracle(self, marked_state :int):
        if marked_state<0 or marked_state>=self.dim:
            raise ValueError("incorrect marked state")
        for q in range(self.n_qubits):
            v=(marked_state>>q) & 1
            if v==0:
                self.gate_1q(self.X, q)

        self.gate_mcz()

        for q in range(self.n_qubits):
            v=(marked_state>>q) & 1
            if v==0:
                self.gate_1q(self.X, q)

    def diffusion(self):
        for q in range(self.n_qubits):
            self.gate_1q(self.H, q)
        for q in range(self.n_qubits):
            self.gate_1q(self.X, q)
        self.gate_mcz()
        for q in range(self.n_qubits):
            self.gate_1q(self.X, q)
        for q in range(self.n_qubits):
            self.gate_1q(self.H, q)

    def grover_step(self, marked_state :int):
        self.oracle(marked_state)
        self.diffusion()


    def norm(self):
        return cp.linalg.norm(self.state).item()

    def probabilities(self):
        return cp.abs(self.state)**2

    def measure_most_probable(self):
        probs=self.probabilities()
        return int(cp.argmax(probs).item())

    def reset(self):
        self.state.fill(0)
        self.state[0]=1.0+0.0j

    def destroy(self):
        if self.handle is not None:
            custatevec.destroy(self.handle)
            self.handle=None
        self.workspace_1qubit=None
        self.workspace_mcz=None

    def console_print(self):
        print(self.state)



