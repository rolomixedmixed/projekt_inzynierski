import cupy as cp
import cuquantum
from cuquantum.bindings import custatevec as custatevec
import numpy as np

class GroverAlgorithm:
    def __init__(self, m:int):
        if m<2:
            raise ValueError("not enough qubits")
        self.m=m
        self.n_qubits=m+5
        self.dim=1<<self.n_qubits

        self.reg_a=list(range(m))
        self.reg_b=[m, m+1]
        self.reg_c=[m+2, m+3]
        self.reg_d=[m+4]

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

        self.Ud=cp.asarray(self.generate_Ud(), dtype=cp.complex64)

        self.handle=custatevec.create()

        self.workspace_1qubit=self.alloc_workspace(self.H, 1, self.m)

        self.workspace_mcz=self.alloc_workspace(self.Z, 1, self.m-1)

        self.workspace_Ud=self.alloc_workspace(self.Ud, 5, 0)

    def __str__(self):
        return self.state.__str__()

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

    def apply_matrix(self, matrix, target, control, control_values, workspace):

        _, workspace_ptr, workspace_size=workspace

        if control is None or len(control)==0:
            control = 0
            control_values = 0
            n_control = 0
        else:
            n_control = len(control)

        custatevec.apply_matrix(
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

    def generate_Ud(self):
        matrix=cp.zeros((2**5, 2**5), dtype=cp.complex64)

        for d in range(2):
            for c in range(4):
                for b in range(4):
                    in_idx= (d<<4) | (c<<2) | b
                    out_d = 1-d if abs(c-b)==2 else d
                    out_idx=(out_d<<4)|(c<<2)|b
                    matrix[out_idx, in_idx]=1.0

        return matrix

    def bc_matrix(self, num_nodes, quadrants, target_reg):

        for i in range(num_nodes):
            q_val=quadrants[i]
            if q_val==0:
                continue

            control_vals=[(i>>k) & 1 for k in range(self.m)]

            if q_val in [1,3]:
                self.gate_1q(self.X, target_reg[0], self.reg_a, control_vals)
            if q_val in [2,3]:
                self.gate_1q(self.X, target_reg[1], self.reg_a, control_vals)


    def gate_1q(self, matrix, target, control=None, control_values=None):
        if target<0 or target >= self.n_qubits:
            raise ValueError("incorrect target qubit")
        self.apply_matrix(matrix, [target], control, control_values, self.workspace_1qubit)

    def gate_mcz(self):
        target=[self.m-1]
        control=list(range(self.m-1))
        control_values=[1]*(self.m-1)
        self.apply_matrix(self.Z, target, control, control_values, self.workspace_mcz)

    def applyH_n(self):
        for q in range(self.m):
            self.gate_1q(self.H, q)

    def oracle(self, num_nodes, quadrants_b, quadrants_c):
        self.bc_matrix(num_nodes, quadrants_b, self.reg_b)
        self.bc_matrix(num_nodes, quadrants_c, self.reg_c)

        targets_Ud=self.reg_b+self.reg_c+self.reg_d
        self.apply_matrix(self.Ud, targets_Ud, [], [], self.workspace_Ud)

        self.bc_matrix(num_nodes, quadrants_b, self.reg_b)
        self.bc_matrix(num_nodes, quadrants_c, self.reg_c)

    def diffusion(self):
        for q in range(self.m):
            self.gate_1q(self.H, q)
        for q in range(self.m):
            self.gate_1q(self.X, q)
        self.gate_mcz()
        for q in range(self.m):
            self.gate_1q(self.X, q)
        for q in range(self.m):
            self.gate_1q(self.H, q)

    def probabilities(self):
        probs=cp.abs(self.state)**2
        a_size=1<<self.m
        probs=probs.reshape(-1, a_size)
        return cp.sum(probs, axis=0)

    def run_for_direction(self, num_nodes, quadrants_b, quadrants_c):
        self.state.fill(0)
        self.state[0]=1.0+0.0j

        self.applyH_n()

        self.gate_1q(self.X, self.reg_d[0])
        self.gate_1q(self.H, self.reg_d[0])

        max_iter=int(np.floor(np.pi*np.sqrt(1<<self.m)/4.0))
        max_iter=max(1, max_iter)

        t=np.random.randint(1, max_iter+1)
        for _ in range(t):
            self.oracle(num_nodes, quadrants_b, quadrants_c)
            self.diffusion()

        probs_a=self.probabilities()
        shots=1024
        sampled_indices=cp.random.choice(1<<self.m, size=shots, p=probs_a)
        counts=cp.bincount(sampled_indices, minlength=(1<<self.m))

        return counts.get(), 1<<self.m


    def reset(self):
        self.state.fill(0)
        self.state[0]=1.0+0.0j

    def destroy(self):
        if self.handle is not None:
            custatevec.destroy(self.handle)
            self.handle=None
        self.workspace_1qubit=None
        self.workspace_mcz=None


def grover_one_dir(edges_dir, global_quadrants, direction):
    num_nodes=len(edges_dir)
    if num_nodes==0:
        return []

    m=int(np.ceil(np.log2(num_nodes)))
    if m<2:
        m=2

    quadrants_b=np.zeros(num_nodes, dtype=np.int32)
    quadrants_c=np.zeros(num_nodes, dtype=np.int32)

    for i, (idx_a,idx_b) in enumerate(edges_dir):
        quadrants_b[i]=global_quadrants[idx_a]
        quadrants_c[i]=global_quadrants[idx_b]

    grover=GroverAlgorithm(m)
    counts, a_size=grover.run_for_direction(num_nodes, quadrants_b, quadrants_c)
    grover.destroy()

    o_max=np.max(counts)
    if o_max==0:
        return []

    candidate_indices=[]
    for i in range(a_size):
        if counts[i]/o_max>0.5:
            if i<num_nodes:
                candidate_indices.append(i)

    ambiguity = len(candidate_indices)/a_size
    if ambiguity >= 0.33:
        print("direction: "+str(direction)+"rejected (too much ambiguity: >=33%)")
        return []

    true_candidates=[edges_dir[i] for i in candidate_indices]

    print("direction: "+str(direction)+" found "+str(len(true_candidates))+" candidates")

    return true_candidates
